import torch
import torch.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,voc_size,hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(voc_size,hidden_size)
        self.linear = nn.Linear(hidden_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
    def initHidden(self):
        return (torch.random((1,hidden_size,hidden_size)),
                torch.random((1, hidden_size, hidden_size)))

    def forward(self,input_,hidden):
        embed = self.embed(input_)
        x = F.relu(self.linear(embed)).unsqueeze(0)
        o,hidden = self.lstm(x,hidden)
        return o,hidden

class Decoder(nn.Module):

