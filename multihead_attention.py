import torch
from torch import nn
import torch.functional as f
import math
d_modal = 512
n_head = 8
X=torch.randn(128,64,512)
class Multihead_attention(nn.Module):
    def __init__(self,d_modal,n_head):
        super(Multihead_attention,self).__init__()
        self.n_head = n_head
        self.d_modal = d_modal
        self.w_q = nn.Linear(d_modal,d_modal)
        self.w_k = nn.Linear(d_modal,d_modal)
        self.w_v = nn.Linear(d_modal,d_modal)
        self.w_combine = nn.Linear(d_modal,d_modal)
        self.softmax = nn.Softmax(dim=-2) #α1,1,α1，2,.. for each column add to 1(given q,soft max q*v1,2,..)
    def forward(self,q,k,v): #in some cases, q,k can be different modalities
        batch, time, dimension = q.shape
        n_d = dimension//self.n_head
        q,k,v = self.w_q(q),self.w_k(q),self.w_v(q)
        q = q.view(batch,time,self.n_head,n_d).permute(0,2,1,3) #(batch,n_head,time,n_d)
        k = k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v = v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score = q @ k.transpose(2,3)/math.sqrt(n_d) # (batch,n_head,time,time)
        mask = torch.tril(torch.ones(time,time,dtype=bool))
        score = score.masked_fill(mask == 0,float("-inf"))
        score = self.softmax(score) @ v  #(batch,n_head,time,n_d)
        score = score.permute(0,2,1,3).contiguous().view(batch,time,dimension) #(batch,size,dimension)
        output = self.w_combine(score)
        return output
Mha = Multihead_attention(d_modal,n_head)
output = Mha(X,X,X)
print(output.shape)