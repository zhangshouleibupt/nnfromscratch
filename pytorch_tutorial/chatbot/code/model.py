import torch
import torch.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,voc_size,hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(voc_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
    def initHidden(self):
        return (torch.random((1,hidden_size,hidden_size)),
                torch.random((1, hidden_size, hidden_size)))

    def forward(self,input_,hidden):
        embed = self.embed(input_).unsqueeze(0)
        o,hidden = self.lstm(embed,hidden)
        return o,hidden

class Decoder(nn.Module):
    def __init__(self,voc_size,hidden_size):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(voc_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,voc_size)
        self.softmax = nn.Softmax(dim=1)
    def initHidden(self):
        return (torch.random(1,self.hidden_size,self.hidden_size),
                torch.random(1,self.hidden_size,self.hidden_size))

    def forward(self,input_,hidden):
        embed = self.embed(input_).unsqueeze(0)
        out,hidden = self.lstm(embed,hidden)
        out = self.out(out).unsqueeze(0)
        out = self.softmax(out)
        return out,hidden
