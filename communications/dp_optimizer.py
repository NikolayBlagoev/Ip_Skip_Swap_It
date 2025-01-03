from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch import no_grad, cat, zeros_like, split, mean
from time import time
from torch.distributed import barrier, all_reduce, ReduceOp
class DP_optim(object):
    def __init__(self, lr, model, dp_group, device):
        super().__init__()
        self.dp_group = dp_group
        self.iteration = 0
        self.net = model
        self.sizes = []
        self.device = device
        self.lr = lr
        self.len_sizes = []
        for param in self.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))
        


    def step(self):
        tmp = []
        t1 = time()
        with no_grad():
            if self.iteration == 0:
                clip_grad_norm_(self.net.parameters(), 1)
                for i, param in enumerate(self.net.parameters()):
                    if param.grad == None:
                        continue
                    param -= self.lr*param.grad
                    param.grad = None
                for param in self.net.parameters():
                    if param == None:
                        tmp.append(zeros_like(param).view(-1))
                                    
                        continue
                    tmp.append(param.view(-1))
            else:
                for param in self.net.parameters():
                    if param.grad == None:
                        tmp.append(zeros_like(param).view(-1))
                                    
                        continue
                    tmp.append(param.grad.view(-1))
        prev_grad = cat(tmp).to("cpu")
        sz = len(prev_grad)*8
        # print("GRADIENT MEAN BEFORE", mean(prev_grad))
        barrier(self.dp_group.group)
        all_reduce(prev_grad, op = ReduceOp.SUM, group=self.dp_group.group)
        # print("GRADIENT MEAN AFTER", mean(prev_grad/4))
        tmp = split(prev_grad, self.len_sizes)
        with no_grad():
            if self.iteration == 0:
                for i, param in enumerate(self.net.parameters()):
                    param = tmp[i].view(self.sizes[i]).to(self.device)/self.dp_group.g_size
            else:
                for i, param in enumerate(self.net.parameters()):
                    param.grad = tmp[i].view(self.sizes[i]).to(self.device)/4
                    # param.grad = None
                clip_grad_norm_(self.net.parameters(), 1)
                for i, param in enumerate(self.net.parameters()):
                    param -= self.lr*param.grad
                    param.grad = None
        self.iteration += 1
        t2 = time()
        self.dp_group.compensate(t1,t2,sz)
        


