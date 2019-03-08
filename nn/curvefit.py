import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
#a polynomial curve fitting 
class CurveFitting():
	def __init__(self,x_set,y_set,M=100):
		self.M = M
		self.x_set = x_set
		self.y_set = y_set
		self.w = Variable(torch.randn(M,1),requires_grad=True)
		#self.loss = feeddata()
	def feeddata(self):
		loss = 0
		for x,y in zip(self.x_set,self.y_set):
			x_boost = self.custome(x)
			out = x_boost.mm(self.w)
			current_loss = (out - y).pow(2).sum()/2
			loss += current_loss
		return loss
	def custome(self,x):
		x = [torch.tensor(x).pow(i).item() for i in range(self.M)]
		return torch.tensor(x).unsqueeze(0)
		
	def gradientDescent(self,lr=0.01,epochs=100):
		loss = self.feeddata()
		for epoch in range(epochs):
			loss.backward()
			self.w.data -= lr * self.w.grad.data
			loss = self.feeddata()
	def predict(self,x):
		x = self.custome(x)
		pred = x.mm(self.w)
		return pred.item()
		

class CurveFit(nn.Module):
	def __init__(self,M=16):
		super(CurveFit,self).__init__()
		self.M = M
		self.l1 = nn.Linear(1,M)
		self.l2 = nn.Linear(M,M)
		self.l3 = nn.Linear(M,M)
		self.l4 = nn.Linear(M,1)
	def forward(self,x):
		x = torch.tensor([x],dtype=torch.float32)
		x = x.unsqueeze(0)
		o = torch.tan(self.l1(x))
		o = torch.tan(self.l2(o))
		o = torch.tan(self.l3(o))
		o = torch.tan(self.l4(o))
		return o
#prepare the data use
seq_length = 50
data_time_steps = np.linspace(-1, 1, seq_length + 1)
data = np.sin(2*math.pi*data_time_steps)

cf = CurveFitting(data_time_steps,data)
cf.feeddata()
cf.gradientDescent()
model = CurveFit()
optimizer = optim.SGD(model.parameters(),lr=0.01)
epochs = 100

def train(model,x_set,y_set,optim):
	all_loss = 0
	for x,y in zip(x_set,y_set):
		optim.zero_grad()
		pred = model.forward(x)
		loss = (pred-y).pow(2).sum()/2
		all_loss += loss
		loss.backward()
		optim.step()
	return all_loss.item()/len(x_set)
loss_buf = []
for epoch in range(epochs):
	loss_buf.append(train(model,data_time_steps,data,optimizer))
pred_data = [model.forward(x).resize(1).item() for x in data_time_steps]

pred_data = [cf.predict(x) for x in data_time_steps]
plt.plot(data,color='red',linewidth=2.0,linestyle='--')
plt.plot(pred_data,color='blue')
plt.show()


