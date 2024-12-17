from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.skipllama.llamabase import *

from torch import optim, save, no_grad
import random
import torch.nn.functional as F
import torch

from torch import nn



class CustomSeq(nn.Sequential):
    def forward(self, *inputs):
        x, start_p, mask, exec_order, crashed = inputs
        self.outputs = {}
        

        
        ml  = list(self._modules.values())
        x = ml[0](x, start_p, mask) 
        self.outputs[0] = x
        self.intermediate_gradients = []
        x = x.detach().clone()
        for idx,i in enumerate(exec_order):
            if idx == 0:
                self.intermediate_gradients.append(None)
                continue

            
            
            x.requires_grad = True
            x.retain_grad()
            tmp = x
            x = ml[i](x, start_p, mask)
            x.retain_grad()
            with torch.no_grad():
                self.intermediate_gradients.append(torch.div(tmp,x) + 1)
            self.outputs[i] = x

            x = x.detach().clone()
        x.requires_grad = True
        x.retain_grad()
        return x
    
    def custom_backward(self,grad,exec_order,crashed):
        idx = len(exec_order) - 1
        # print(idx)
        
        while idx >= 0:

            i = exec_order[idx]
            if i in crashed:
                grad = self.intermediate_gradients[idx] * grad
            else:
                self.outputs[i].backward(grad)
                grad = self.outputs[i].grad.detach().clone()
            idx -= 1
        self.outputs.clear()
        
        



class LLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embedding = LLamaEmbedding(vocab_size,dmodel,padding_idx=padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = CustomSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.ln = nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x, exec_order,crashed=[]):
        _, seq_l = x.shape
        h = self.embedding(x)
        
        h = self.transformers(h,0,None,exec_order,crashed)
        self.tmp = h
        h = self.norm(h)
        output = self.ln(h).float()
        return output
    def custom_backwrad(self, loss, exec_order, crashed = []):
        loss.backward()
        self.transformers.custom_backward(self.tmp.grad,exec_order,crashed)


from itertools import permutations 




pth_num = 3
f = 0.25
from hungarian_algorithm import algorithm
print(pth_num,f)
seq_l = 128
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 64 // pth_num, seq_l=seq_l)
net = LLama(tkns.vocab_size,dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=16)

op = optim.SGD(net.parameters(),lr=4e-3/pth_num,momentum=0,dampening=0,weight_decay=0,nesterov=False)

lr = 1e-3
for _ in range(10):
    loader = iter(ts) 
    for i in range(8000):
        grad_acc = dict()
        grad_avg = dict()
        loss_hist = []
        crashed = []
        for chunk in range(2,8):

            if random.random() < f:
                
                crashed.append(chunk)
        
        op.zero_grad()
        for p in range(pth_num):
            
            x,y = next(loader)
            x = x.to("cuda")
            y = y.to("cuda")
            
            exec_order = [k for k in range(0,8)]
            to_skip = []
            if i % 10 == 0:
                to_skip = []
            else:
                
                to_skip += [k for k in range((p+1)*2,(p+2)*2)]
                exec_order = [k for k in range(0,8) if k not in to_skip]
         
                        
            path = []
            for k in exec_order:
                path.append(k*2)
                path.append(k*2 + 1)


            # print(exec_order, crashed,to_skip)
               
            
            x = net(x,path)
            B, T, C = x.shape
            x = x.view(B*T,C)
            y = y.view(B*T)
            loss = F.cross_entropy(x,y)
            if i % 10 == 0:
                loss_hist.append(loss.item())
            crashed_new = []
            for k in crashed:
                crashed_new.append(k*2)
                crashed_new.append(k*2 + 1)
            net.custom_backwrad(loss,path,crashed_new)
            
        
        op.step()
        if i%10 == 0:
            print(sum(loss_hist)/len(loss_hist))
        if i%100 == 0:
            save(net.state_dict(), f"back_train_{pth_num}_{f}.pth")

    
    
    
    
   
    

