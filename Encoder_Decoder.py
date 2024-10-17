import torch
from statsmodels.sandbox.regression.ols_anova_original import dropn
from torch import nn
import torch.functional as F
from encoder import LayerNorm,PositionWiseFeed
class Masked_multihead_attention(nn.Module): # ma with mask
    def __init__(self,d_modal,n_head):
        super(Masked_multihead_attention,self).__init__()
        self.n_head = n_head
        self.d_modal = d_modal
        self.w_q = nn.Linear(d_modal,d_modal)
        self.w_k = nn.Linear(d_modal,d_modal)
        self.w_v = nn.Linear(d_modal,d_modal)
        self.w_combine = nn.Linear(d_modal,d_modal)
        self.softmax = nn.Softmax(dim=-2) #α1,1,α1，2,.. for each column add to 1(given q,soft max q*v1,2,..)
    def forward(self,q,k,v,mask=None): #if mask = None then its equal to the original ma
        batch, time, dimension = q.shape
        n_d = dimension//self.n_head
        q,k,v = self.w_q(q),self.w_k(q),self.w_v(q)
        q = q.view(batch,time,self.n_head,n_d).permute(0,2,1,3) #(batch,n_head,time,n_d)
        k = k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v = v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score = q @ k.transpose(2,3)/math.sqrt(n_d) # (batch,n_head,time,time)
        if mask is not None:
            #mask = torch.tril(torch.ones(time,time,dtype=bool))
            score = score.masked_fill(mask == 0,float("-inf"))
        score = self.softmax(score) @ v  #(batch,n_head,time,n_d)
        score = score.permute(0,2,1,3).contiguous().view(batch,time,dimension) #(batch,size,dimension)
        output = self.w_combine(score)
        return output

class Decoder_layer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(Decoder_layer,self).__init__()
        self.attention1 = Masked_multihead_attentionasked(d_model,n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = Masked_multihead_attention(d_model,n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeed(d_model,ffn_hidden)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        _x = dec
        x = self.attention1(dec,dec,dec,t_mask) #lower triangle
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm2(x + _x)
        return x
class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob):
        super(Decoder,self).__init__()
        #
        self.layers = nn.ModuleList(Decoder_layer(d_model,ffn_hidden,n_head,drop_prob) for _ in range(n_layer))
        self.fc = nn.Linear(d_model, dec_voc_size)
    def forward(self,enc,dec,t_mask,s_mask):
        #dec =
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec =self.fc(dec)

        return dec
