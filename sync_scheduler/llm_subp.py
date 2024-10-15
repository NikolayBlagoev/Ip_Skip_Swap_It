from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
import pickle
from torch import manual_seed
from typing import Callable
from contextlib import redirect_stdout
import time
import torch
from torch import optim, stack, mean, split, cat, tensor,save, zeros_like
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.llama import LLamaClassification,LLamaEmbedding,precompute_freqs_cis,TransformerBlock,RMSNorm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch import cuda
import traceback
from simplellm.llama import LLamaStage, LLamaFirstStage, LLamaClassification
import os
import time
from torch.nn import CrossEntropyLoss, Linear

# Messages Exchanged by the processes
@dataclass
class Forward:
    mb: bytes
    tag: bytes


@dataclass
class Backward:
    mb: bytes
    tag: bytes

@dataclass
class Start:
    tag: bytes

@dataclass
class Loss:
    mb: bytes
    tag: bytes


@dataclass
class Aggregate:
    tag: bytes
class LLaMaFS(torch.nn.Module):
    def __init__(self, head, tail):
        super().__init__()

        self.head = head
        self.tail = tail
def run_p(maind_addr,queue_in: Queue, queue_out: Queue, node_id: int = 0, stage: int = 0,
                    device = "cuda"):
    manual_seed(0)
    seq_l = 128
    if node_id == 0:
        tkns = SPTokenizer()
        ts = TinyStories(tkns,batch_size = 32, seq_l=seq_l)
        vals = TinyStories(tkns,batch_size = 32, seq_l=seq_l, split = "validation")
        head = LLamaFirstStage(tkns.vocab_size,dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=4)
        tail = LLamaClassification(tkns.vocab_size,dmodel=256,type="cross_entropy")
        net = LLaMaFS(head,tail)
        optimizer = optim.SGD(net.parameters(),lr=4e-3,momentum=0,dampening=0,weight_decay=0,nesterov=False)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,ts,vals,device=device)
            loc.start()
    else:
        net = LLamaStage(dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=4)
        optimizer = optim.SGD(net.parameters(),lr=4e-3,momentum=0,dampening=0,weight_decay=0,nesterov=False)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,None,None,device=device)
            loc.start()
        
        
    
    
class SubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, stage = 0, ds = None, vals = None, lr = 4e-3,
                    device = "cuda") -> None:
        self.net = net
        self.net.to(device)
        self.device = device
        self.lr = lr
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        self.node_id = node_id
        
        self.iteration = 0
        self.peer_parameters = dict()
        self.prev_aggregation = 0
        self.started = True
        
        self.sizes = []
        self.len_sizes = []
        self.ttl_l = 0
        self.next_gradients = dict()
        self.epoch = 0
        self.buffer_in = {}
        self.buffer_out= {}
        self.optimizer.zero_grad()
        if node_id == 0:
            self.ds = ds
            self.dl = iter(ds)
            self.valds = vals
            self.target = {}
        self.aggregation = []
        self.sizes = []
        self.len_sizes = []
        for param in self.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))
        self.start()
        


    def start(self):
        try:
            while self.started:
                while self.queue_in.empty() and self.started:
                    
                    continue
                if not self.started:
                    break
                task = self.queue_in.get(True)
                if isinstance(task, Start):
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"=======NEW ITERATION:========\n")
                    
                    try:
                        
                        
                        x, y = next(self.dl)
                    except StopIteration:
                        
                        self.epoch += 1
                        self.dl = iter(self.ds)
                        
                        x, y = next(self.dl)
                    with torch.no_grad():
                        x = x.to(self.device)
                        y = y.to(self.device)
                    self.buffer_in[task.tag] = x
                    
                    
                     
                    self.target[task.tag] = y
                    x = self.net.head(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    self.queue_out.put(Forward(pickle.dumps(x),task.tag))
                    continue
                elif isinstance(task, Loss):
                    x = pickle.loads(task.mb)
                    y = self.target[task.tag]
                    loss = self.net.tail(x,y)
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSS:{loss.item()}\n")
                        log.write(f"ITERATION:{self.iteration}\n")
                    loss.backward()
                    self.queue_out.put(Backward(pickle.dumps(x.grad.to("cpu")),task.tag))
                    
                elif isinstance(task, Forward):
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"Forward")
                    x = pickle.loads(task.mb)

                    with torch.no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()
                    self.buffer_in[task.tag] = x
                    x = self.net(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    self.queue_out.put(Forward(pickle.dumps(x),task.tag))
                    continue
                    
                elif isinstance(task, Backward):
                    self.optimizer.zero_grad()
                    
                    output = pickle.loads(task.mb)
                    with torch.no_grad():
                        output = output.to(self.device)
                    inp_batch = self.buffer_out[task.tag]
                    inp_batch.backward(output)
                    if self.node_id != 0:
                        self.queue_out.put(Backward(pickle.dumps(inp_batch.grad.to("cpu")),task.tag),True)
                    
                    tmp = []
        
                    for param in self.net.parameters():
                        if param.grad == None:
                            tmp.append(zeros_like(param).view(-1))
                            
                            continue
                        tmp.append(param.grad.detach().clone().view(-1))
                    prev_grad = cat(tmp).to("cpu")
                    self.aggregation.append(prev_grad)
                    del self.buffer_in[task.tag]
                    del self.buffer_out[task.tag] 
                    cuda.empty_cache()
                    
                
                elif isinstance(task, Aggregate):
                    if len(self.aggregation) == 0:
                        continue
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"===AGGEGATING==== {len(self.aggregation)}\n")
                    tmp = torch.stack(self.aggregation)
                    tmp = torch.mean(tmp, dim = 0).to(self.device)
                    i = 0
                    self.iteration += 1
                    while i < len(self.aggregation):
                        del self.aggregation[i]
                    
                    self.aggregation.clear()
                    
                    
                    ret = split(tmp, self.len_sizes)
                    
                    for i, param in enumerate(self.net.parameters()):
                        param.data -= self.lr*ret[i].view(self.sizes[i]).to(self.device).data
                    del ret
                    del tmp
                    cuda.empty_cache()
        except Exception:
            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                log.write(f"{traceback.format_exc()}\n")
            
            exit()




    def custom_avg(self,list_of_tensors):
        tmp = mean(stack(list_of_tensors), dim = 0)
        i = 0
        while i < len(list_of_tensors):
            del list_of_tensors[i]
        
        return tmp



                


                
        

    
