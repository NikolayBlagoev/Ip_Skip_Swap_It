from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.swapllama import LLama,LLamaClassification
from torch import optim, save, no_grad
import random
import torch.nn.functional as F
from itertools import permutations 
pth_num = 3
f = 0.25
from hungarian_algorithm import algorithm
print(pth_num,f)
# seq_l = 128
# tkns = SPTokenizer()
# ts = TinyStories(tkns,batch_size = 64 // pth_num, seq_l=seq_l)
# net = LLama(tkns.vocab_size,dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=16)

# op = optim.SGD(net.parameters(),lr=4e-3/pth_num,momentum=0,dampening=0,weight_decay=0,nesterov=False)

lr = 1e-3
for _ in range(10):
    # loader = iter(ts) 
    for i in range(8000):
        grad_acc = dict()
        grad_avg = dict()
        loss_hist = []
        crashed = []
        for chunk in range(2,8):

            if random.random() < f:
                
                crashed.append(chunk)
        crashed = [2, 3, 6]
        # op.zero_grad()
        for p in range(pth_num):
            
            # x,y = next(loader)
            # x = x.to("cuda")
            # y = y.to("cuda")
            
            exec_order = [k for k in range(0,8)]
            to_skip = []
            if i % 10 == 1:
                to_skip = []
            else:
                
                to_skip += [k for k in range((p+1)*2,(p+2)*2)]
                exec_order = [k for k in range(0,8) if k not in to_skip]
                bipartite_match = {}
                to_skip = [k for k in to_skip if k not in crashed]
                graph = {}
                added = False
                for t in crashed:
                    for c in to_skip:
                        
                        if t not in graph:
                            graph[t] = {}
                        added = True    
                        if abs(t-c) > 2:
                            graph[t][c] = 9000
                        else:
                            graph[t][c] = abs(t-c)
                        
                

                if added:
                    print(graph)
                    ret = algorithm.find_matching(graph, matching_type = 'min', return_type = 'list' )
                    
                    if  isinstance(ret,list):
                        for r in ret:
                            if r[1] > 10:
                                continue
                            bipartite_match[r[0][0]] = r[0][1]
                
                for k in range(len(exec_order)):
                    if  exec_order[k] in crashed:
                        if exec_order[k] in bipartite_match:
                            exec_order[k] = bipartite_match[exec_order[k]]
                        else:
                            exec_order[k] = 100000
                        
                        
                        
            path = []
            for k in exec_order:
                if k > 10:
                    continue
                path.append(k*2)
                path.append(k*2 + 1)


            print(exec_order, crashed,to_skip)
               
            
            # x = net(x,path)
            # B, T, C = x.shape
            # x = x.view(B*T,C)
            # y = y.view(B*T)
            # loss = F.cross_entropy(x,y)
            # if i % 10 == 0:
            #     loss_hist.append(loss.item())
            # loss.backward()
        exit()
            
        
        # op.step()
        # if i%10 == 0:
        #     print(sum(loss_hist)/len(loss_hist))
        # if i%100 == 0:
        #     save(net.state_dict(), f"swap_train_{pth_num}_{f}.pth")

    
    
    
    
   
    

    


