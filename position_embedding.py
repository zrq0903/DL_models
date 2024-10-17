import torch
from torch import nn
class position_embedding(nn.Module):
    def __init__(self,n_pos_vec,dim):
        super(position_embedding,self).__init__()
        self.encoding = torch.zeros(n_pos_vec,dim) #(n,dim)
        self.encoding.requires_grad = False
        _2i = torch.arange(dim//2,dtype = float)
        _2i /= (dim/2)
        _2i = 1/(10000**_2i) # (dim//2)
        pos = torch.arange(0,n_pos_vec) #(n)
        pos = pos.float().unsqueeze(1) #(n,1)
        out = pos/_2i
        self.encoding[:,0::2]=torch.sin(out)
        self.encoding[:,1::2]=torch.cos(out) #(n,dim)
    def forward(self,x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len,:]# ensuring the positional embeddings are only applied to the valid sequence length.
# to add the word embedding or we can write  return x + self.encoding[:seq_len,:]

class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size, d_modal):
        super(TokenEmbedding,self).__init__(vocab_size,d_modal,padding_idx=1)