import gzip
import numpy as np
from PIL import Image
import random
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
def readLables(file_name):
	file_content = []
	with gzip.open(file_name,mode="rb") as f:
		file_content = f.read()
	file_content = file_content[8:]
	labels = [file_content[i] 
			 for i in range(len(file_content))]
	return labels

def readImages(file_name,numbers=60000):
	file_content = []
	with gzip.open(file_name,mode="rb") as f:
		file_content = f.read()
	file_content = file_content[16:]

	images = []
	for i in range(numbers):
		piexs = file_content[i*28*28:(i+1)*28*28]
		piexs = [piexs[index] for index in range(28*28)]
		image = np.array(piexs,dtype=np.uint8)
		#image = image.reshape(28,28)
		image = image.reshape(1,-1)
		images.append(image)
	return images

def showImage(array_):
	im = Image.fromarray(array_,mode="L")
	im.show()


train_images_file = "../data/train-images-idx3-ubyte.gz"
test_images_file = "../data/t10k-images-idx3-ubyte.gz"
train_labels_file = "../data/train-labels-idx1-ubyte.gz"
test_labels_file = "../data/t10k-labels-idx1-ubyte.gz"
train_set_image = readImages(train_images_file)
train_set_label = readLables(train_labels_file)
test_set_image = readImages(test_images_file,numbers=1000)
test_set_label = readLables(test_labels_file)



def train(images,labels,model,criterion,optim):
	batch_size = 60
	iters = len(images) // batch_size
	loss = 0
	all_loss = 0
	loss_buf = []
	for it in range(iters):
		optim.zero_grad()
		data = images[it*batch_size:batch_size*(it+1)]
		batch_data = torch.randn((batch_size,1,28*28))
		for i in range(batch_size):
			batch_data[i] = torch.tensor(data[i],dtype=torch.float32)
		label = labels[it*batch_size:batch_size*(it+1)]
		label = torch.tensor(label,dtype=torch.long)
		output = model(batch_data).squeeze(1)
		loss = criterion(output,label)
		loss.backward()
		optim.step()
		all_loss += loss
		if it % 10 == 0:
			loss_buf.append(all_loss.item()/600)
			all_loss = 0
		if it % 100 == 0:
			print('have finished % {}'.format((it//100+1)*10))
	return loss_buf
input_size = 28*28
output_size = 10
net = model.NeuralNet(input_size,output_size)

loss_buf = []
epochs = 10
lr = 0.01
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(),lr=lr)
for epoch in range(epochs):
	print("in epoch {}:".format(epoch))
	loss_buf += train(train_set_image,train_set_label,net,criterion,optimizer)
plt.plot(loss_buf)
plt.show()	

def isPredictRight(out,y):
	if out.argmax(dim=2) == y:
		return 1
	else:
		return 0
acc = 0.0
all_items = len(test_set_image)
for x,y in zip(test_set_image,test_set_label):
	x = torch.tensor(x,dtype=torch.float32).view(1,1,-1)
	out = net(x)
	acc += isPredictRight(out,y)

print('the acc is {}'.format(acc/all_items))
