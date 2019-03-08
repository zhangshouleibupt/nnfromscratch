import torch
from torch.autograd import Variable
import numpy as np
import math
import matplotlib.pyplot as plt
class RNN():
	def __init__(self,input_size,hidden_size,output_size):
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size
		self.w1,self.w2 = self.initParameter()
	
	def initParameter(self):
		w1 = torch.randn(input_size+hidden_size,hidden_size)
		w2 = torch.randn(hidden_size,output_size)
		w1,w2 = Variable(w1,requires_grad=True),Variable(w2,requires_grad=True)
		return w1,w2
	def initHidden(self):
		return torch.randn(1,self.hidden_size)
	def forward(self,input_,hidden):
		combined = torch.cat((input_,hidden),dim=1)
		hidden = torch.tanh(combined.mm(self.w1))
		output = hidden.mm(self.w2)
		return output,hidden
	def backpro(self,lr=lr):
		self.w1.data -= self.w1.grad.data * lr
		self.w2.data -= self.w2.grad.data *lr
		self.w1.grad.data.zero_()
		self.w2.grad.data.zero_()
		

input_size = 1
hidden_size = 6
output_size = 1
dtype = torch.FloatTensor
lr = 0.01

t = np.linspace(2,10,500+1)
data = np.sin(t)

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)
#plt.plot(data,color='blue',linestyle = "-")
#plt.show()
rnn = RNN(input_size,hidden_size,output_size)
def train(model,x_set,y_set):
	hidden = model.initHidden()
	all_loss = 0
	for x,y in zip(x_set,y_set):
		x = torch.tensor([x],dtype=torch.float32).unsqueeze(0)
		out,hidden = model.forward(x,hidden)
		loss = (out-y).pow(2).sum()/2
		loss.backward()
		all_loss += loss
		model.backpro()
		hidden = Variable(hidden)
	return all_loss.item()/len(x_set)
loss_buf = []
for epoch in range(epochs):
	loss_buf.append(train(rnn,x,y))
hidden = rnn.initHidden()
pred = []
#predict 
for t in x:
	t = torch.tensor([t],dtype=torch.float32).unsqueeze(0)
	out,hidden = rnn.forward(t,hidden)
	pred.append(out.unsqueeze(0).item())
plt.plot(pred,color='red',linewidth=2.0,linestyle="--")
plt.plot(data,color="blue")
plt.savefig('./rnn_fit.png')
plt.show()

