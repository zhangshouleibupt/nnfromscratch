import pickle
import torch
import random
from dataload import VOC

voc_file = "../data/voc.pkl"
file_path = '../data/conversation_pais.txt'
content = []
with open(file_path,encoding='utf-8') as f:
    content = [line.split('+$+') for line in f]
voc = []
with open(voc_file,'rb') as f:
    voc = pickle.load(f)
sample = random.choice(content)
print(sample)
def sentenceToTensor(sentence,voc):
    dtype = torch.long
    sen_list = [voc.word2Index[word] for word in sentence.rstrip('\n').split(' ')]
    sen_ten = torch.tensor(sen_list,dtype=dtype)
    return sen_ten

for p in sample:
    print(sentenceToTensor(p,voc))