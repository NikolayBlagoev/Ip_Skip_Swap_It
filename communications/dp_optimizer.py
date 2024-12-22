from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch import no_grad, cat
from torch.distributed import barrier, all_reduce, ReduceOp
class DP_optim(object):
    def __init__(self, lr, model, dp_group):
        super().__init__()
        self.dp_group = dp_group
        self.iteration = 0
        self.net = model
        self.sizes = []
        self.lr = lr
        self.len_sizes = []
        for param in self.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))
        


    def step(self):
        tmp = []
        
        for param in self.net.parameters():
            if param.grad == None:
                tmp.append(zeros_like(param.data).view(-1))
                            
                continue
            tmp.append(param.grad.view(-1))
        prev_grad = cat(tmp).to("cpu")

        barrier(self.dp_group)
        all_reduce(prev_grad, op = ReduceOp.AVG, group=self.dp_group)
        # TODO CLIP IT TO 1.0!!!
        tmp = split(prev_grad, self.len_sizes)
        with no_grad():
            for i, param in enumerate(self.net.parameters()):
                param = param - self.lr*tmp[i].view(self.sizes[i]).to(self.net.device)
                param.grad = None
        self.iteration += 1
        


