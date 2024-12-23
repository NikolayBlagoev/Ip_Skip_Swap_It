from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.llama.llama import SwapLLama,LLama,SkipLLama
from torch import optim, save, no_grad
import random
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from copy import deepcopy
from simplellm.losses import causalLLMLoss
pth_num = 6
f = 0.00

print(pth_num,f)
seq_l = 256
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 96 // pth_num, seq_l=seq_l)
net = LLama(SkipLLama,tkns.vocab_size,dmodel=288,num_heads=6,multiple_of=32,ctx_size=seq_l,n_layers=16)
vocab_size = tkns.vocab_size
random.seed(10)
lr = 1e-3
lrs = {}

can_crash  = False
for iteration in range(20):
    loader = iter(ts.dl) 
    for i in range(1000):
        loss_hist = []
        crashed = []
        for chunk in range(2,8):

            if random.random() < f:
                crashed += [k for k in range(chunk*2,(chunk+1)*2)]
        
        
        for p in range(pth_num):
            
            
            x,y = next(loader)
            x = x.to("cuda")
            y = y.to("cuda")
            to_skip = []
            
            #print(exec_order,to_skip,crashed)
            x = net(x,0,None, None,[])
            B, T, C = x.shape
            x = x.view(B*T,C)
            y = y.view(B*T)
            loss = F.cross_entropy(x,y,ignore_index=-1)
            if i % 10 == 0:
                loss_hist.append(loss.item())
            loss = loss / pth_num
            loss.backward()
            
            
        clip_grad_norm_(net.parameters(), 1)
        lr = 4e-3 / (1+(iteration//4))
        with no_grad():
            for p in net.parameters():
                if p.grad == None:
                    continue
                p -= p.grad*lr
                p.grad = None


        if i%10 == 0:
            print(sum(loss_hist)/len(loss_hist))
        if i%100 == 0:
            save(net.state_dict(), f"download_train_{pth_num}_{f}.pth")

    
