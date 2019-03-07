import torch
from torch.autograd import Variable

t = Variable(torch.randn(1,2),requires_grad=True)

epochs = 1000
lr = 0.01
for epoch in range(epochs):
	y = torch.sum(t**2)/2
	y.backward()
	t.data -= lr * t.grad.data
	print(t)
	if y.item() < 0.1:
		lr = 0.001
		print(lr)
