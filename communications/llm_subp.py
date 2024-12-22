from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
from torch import manual_seed
from torch.distributed import recv, isend, irecv, send
from torch import Tensor, zeros, save
from simplellm.tokenizers import SPTokenizer
from simplellm.llama import LLamaFirstStage, LLamaStage
from .dp_group import DP_Group, initialise_communication
from .dp_optimizer import DP_optim
from torch import cuda, no_grad
import traceback
from contextlib import redirect_stdout
from simplellm.dataloaders import Wikipedia_Dataset
import torch.functional as F
from time import time, sleep
# Messages Exchanged by the processes
@dataclass
class Forward:
    tag: int
    frm: int
    to: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor
@dataclass
class Backward:
    tag: int
    frm: int
    to: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor

@dataclass
class Start:
    tag: int
    to: int
    originator: int
@dataclass
class Deferred:
    tag: int
@dataclass
class Loss:
    tag: int
    frm: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor



@dataclass
class Aggregate:
    epoch: int

def run_p(main_addr, partitions, queue_in: Queue, queue_out: Queue, node_id: int = 0, stage: int = 0, seq_l: int = 256, n_layers = 4, 
                    batch_size = 8, dmodel = 256, multiple_of = 4, num_heads = 16, memory = 3, process_time = 2,
                    device = "cuda"):
    manual_seed(0)
    world_size = 0
    for v in partitions:
        world_size += len(v)
    group = initialise_communication(partitions,node_id, main_addr, world_size)
    
    if stage == 0:
        tkns = SPTokenizer()
        ts = Wikipedia_Dataset(tkns,batch_size = batch_size, seq_l=seq_l)
        vals = Wikipedia_Dataset(tkns,batch_size = batch_size, seq_l=seq_l, split = "train")
        net = LLamaFirstStage(tkns.vocab_size, dmodel, num_heads, n_layers, multiple_of = multiple_of, ctx_size= seq_l)
        
        optimizer = DP_optim(4e-3, net, group)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,ts,vals,device=device, memory = memory,process_time=process_time)
            loc.start()
    else:
        net = LLamaStage(ctx_size=seq_l, dmodel=dmodel,num_heads=num_heads,multiple_of=multiple_of,n_layers=n_layers)
        optimizer = DP_optim(4e-3, net, group)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,None,None,device=device, memory = memory,process_time=process_time)
            loc.start()


class SubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, stage = 0, ds = None, vals = None, 
                    device = "cuda", mb_count = 12, process_time = 2, memory = 3) -> None:
        self.net = net
        self.process_time = process_time
        self.memory = memory
        self.MAX_MEM = memory
        self.net.to(device)
        self.device = device
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        self.node_id = node_id
        self.buffer_in = {}
        self.buffer_out = {}
        self.receives = []
        self.iteration = 0
        
        self.started = True
        self.deferred = {}
        
        self.epoch = 0
        
        if stage == 0:
            self.ds = ds
            self.dl = iter(ds)
            self.valds = vals
            self.target = {}
        
        
        


    def start(self):
        try:
            while self.started:
                task = None
                while self.queue_in.empty() and self.started and task == None:
                    for idx,el in enumerate(self.receives):
                        if el[1].done():
                            task = el[0]
                            self.receives.pop(idx)
                            break

                    continue
                if not self.started:
                    break
                if task == None:
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
                    with no_grad():
                        x = x.to(self.device)
                        
                    self.buffer_in[task.tag] = x
                    
                    
                    
                    self.target[task.tag] = y
                    tm1 = time()
                    x = self.net.embed(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    tm2 = time()
                    if tm2 - tm1 < self.process_time:
                        sleep(self.process_time - (tm2 - tm1))
                    ret = x.to("cpu")
                    
                    
                    self.queue_out.put(Forward(task.tag, self.node_id, task.to, x.shape[0], x.shape[1], x.shape[2], task.originator, None), True)
                    
                    if self.iteration == 0:
                        isend(ret,task.to).wait() # First iteration we need to block :))
                    else:
                        isend(ret,task.to)
                    
                    continue
                elif isinstance(task, Loss):
                    if task.data == None:
                        x = zeros((task.B,task.T,task.C))
                        if self.iteration == 0:
                            irecv(x,task.frm).wait()
                            task.data = x
                        else:
                            task.data = x
                            self.receives.append((task,irecv(x,task.frm)))
                            continue
                    
                    x = task.data
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()
                    y = self.target[task.tag].to(self.device)
                    tm1 = time()
                    ret = self.net.forward_end(x)
                    
                    B, T, C = ret.shape
                    ret = ret.view(B*T,C)
                    y = y.view(B*T)
                    loss = F.cross_entropy(ret,y)
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSS:{loss.item()}\n")
                        log.write(f"ITERATION:{self.iteration}\n")
                    loss = loss / self.mb_count
                    
                    loss.backward()
                    tm2 = time()
                    if tm2 - tm1 < 2*self.process_time:
                        sleep(2*self.process_time - (tm2 - tm1))
                    ret = x.grad
                    ret = ret.to("cpu")
                    send = isend(ret.grad, task.frm)
                    self.queue_out.put(Backward(task.tag, task.frm, task.to, x.grad.shape[0], x.grad.shape[1], x.grad.shape[2], task.originator, None), True)
                    if self.iteration == 0:
                        send.wait()
                elif isinstance(task, Forward):
                    #TODO: Need to receive asynchonously after 1st round
                    if task.data == None:
                        x = zeros((task.B,task.T,task.C))
                        if self.iteration == 0:
                            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                                log.write(f"TO RECEIVE FROM {task.frm}\n")
                            recv(x,task.frm)
                            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                                log.write(f"RECEIVED FROM {task.frm}\n")
                            task.data = x
                        else:
                            task.data = x
                            self.receives.append((task,irecv(x,task.frm)))
                            continue
                    
                    if task.tag not in self.deferred:
                        
                        
                        
                        if self.memory == 0:
                            self.deferred[task.tag] = (x,task)
                            self.queue_out.put(Deferred(task.tag), True)
                            continue
                    else:
                        
                        task = self.deferred[task.tag][1]
                        del self.deferred[task.tag]
                    x = task.data
                    self.memory -= 1
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()

                    self.buffer_in[task.tag] = x
                    tm1 = time()
                    x = self.net(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    tm2 = time()
                    if tm2 - tm1 < self.process_time:
                        sleep(self.process_time - (tm2 - tm1))
                    ret = x.to("cpu")
                    send = isend(ret,task.to)
                    self.queue_out.put(Forward(task.tag, task.frm, task.to, x.shape[0], x.shape[1], x.shape[2], task.originator, None), True)
                    if self.iteration == 0:
                        send.wait()
                        with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                            log.write(f"Finished sending to {task.to} {x.shape[0]} {x.shape[1]} {x.shape[2]}\n")
                    continue
                    
                elif isinstance(task, Backward):
                    #TODO: Need to receive asynch > first round
                    if task.data == None:
                        x = zeros((task.B,task.T,task.C))
                        if self.iteration == 0:
                            irecv(x,task.frm).wait()
                            task.data = x
                        else:
                            task.data = x
                            self.receives.append((task,irecv(x,task.frm)))
                            continue
                    output = task.data
                    # irecv(output,task.frm)
                    tm1 = time()
                    with no_grad():
                        output = output.to(self.device)
                    inp_batch = self.buffer_out[task.tag]
                    
                    inp_batch.backward(output)
                    tm2 = time()
                    if tm2 - tm1 < 2*self.process_time:
                        sleep(2*self.process_time - (tm2 - tm1))
                    self.memory += 1
                    if task.frm != -1:
                        ret = inp_batch.grad
                        ret = ret.to("cpu")
                        isend(ret, task.to)

                        self.queue_out.put(Backward(task.tag, task.frm, task.to, inp_batch.grad.shape[0], inp_batch.grad.shape[1], inp_batch.grad.shape[2], task.originator, None),True)
                    
 
                    del self.buffer_in[task.tag]
                    del self.buffer_out[task.tag] 
                    cuda.empty_cache()
                    
                
                elif isinstance(task, Aggregate):
                    assert self.memory == self.MAX_MEM
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"===AGGEGATING====\n")
                    self.buffer_in.clear()
                    self.buffer_out.clear()
                    cuda.empty_cache()
                    self.iteration += 1

                    # update params
                    self.optimizer.step() # this also syncs across stage

                    # save model
                    if self.iteration % 2000 == 0:
                       save(self.net.state_dict(), f"{(self.epoch//2000) % 5}_{self.node_id}.pth") 
                    cuda.empty_cache()
                    self.queue_out.put(Aggregate(0), True)
        except Exception:
            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                log.write(f"{traceback.format_exc()}\n")
            
            exit()