import torch
from torch import nn
import torch.functional as F
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-10):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = torch.mean(x,-1,keepdim=True)
        var = torch.var(x,-1,unbiased=False,keepdim=True)
        x = (x-mean)/torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x
class PositionWiseFeed(nn.Module):
    def __init__(self,d_model,hidden):
        super(PositionWiseFeed,self).__init__()
        self.fc1 = nn.Linear(d_model,hidden)
        self.fc2 = nn.Linear(hidden,d_model)
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
