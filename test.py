import torch

x = torch.tensor([[1.0,2.0],[1.0,3.0]])
x.requires_grad = True
y = torch.tensor([[1.0,1.0],[4.0,2.0]])
out = x*y
out.backward(gradient=torch.tensor([[1.0,1.0],[1.0,1.0]]))

print(x.grad)
print(y.grad)
print(torch.ones_like(x))