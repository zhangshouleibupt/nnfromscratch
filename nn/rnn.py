import torch
from torch.autograd import Variable


# standard gradient descent
t = Variable(torch.randn(1,2),requires_grad=True)
epochs = 1000
lr = 0.01
def gradientDescent():
	for epoch in range(epochs):
		y = torch.sum(t**2)/2
		y.backward()
		t.data -= lr * t.grad.data
		#print(t)
		if y.item() < 0.1:
			lr = 0.001
#cofigure one layer rnn
#the standard rnn model formula
#h_t = f(x_t,h_(t-1))
#the variety of rnn used to simulate 
#the sine wave
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
		hidden = combined.mm(self.w1)
		output = torch.tanh(hidden.mm(self.w2))
		return output,hidden
	def backpro(self,lr=lr):
		self.w1.data -= self.w1.grad.data * lr
		self.w2.data -= self.w2.grad.data *lr
input_size = 1
hidden_size = 6
output_size = 1
lr = 0.01

