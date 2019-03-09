import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
	
	def __init__(self,output_size):
		super(ConvNet,self).__init__()
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*4*4,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,output_size)
	
	def forward(self,input_):
		x = F.relu(self.conv1(input_))
		x = F.max_pool2d(x,(2,2))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x,2)
		x = x.view(-1,1,self.getfeaturesize(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		o = F.log_softmax(x,dim=2)
		return o
	def getfeaturesize(self,x):
		size = x.size()[1:]
		features = 1
		for s in size:
			features *= s
		return features
class NeuralNet(nn.Module):
	
	def __init__(self,input_size,output_size):
		super(NeuralNet,self).__init__()
		self.l1 = nn.Linear(input_size,128)
		self.l2 = nn.Linear(128,64)
		self.l3 = nn.Linear(64,32)
		self.l4 = nn.Linear(32,output_size)
		
	
	def forward(self,input_):
		x = torch.tanh(self.l1(input_))
		x = torch.relu(self.l2(x))
		x = torch.tanh(self.l3(x))
		x = torch.relu(self.l4(x))
		out = F.log_softmax(x,dim=2)
		return out
	
