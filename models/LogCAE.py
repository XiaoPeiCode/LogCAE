
import torch
import torch.nn as nn

from models.AutoEncoder import AE
from models.Emp2Rep import Emb2Rep


class LogCAE(nn.Module):
    def __init__(self,input_dim=768,rnn_hidden_dim=512,rep_dim=128,ae_hidden=[256,128,64],num_heads=8):
        super(LogCAE, self).__init__()

        #Emb2Rep
        self.emb2rep = Emb2Rep(input_dim=input_dim, hidden_dim=rnn_hidden_dim, output_size=rep_dim,
                        num_heads=num_heads)
        # 创建模型rep
        # AE
        self.ae = AE(input_dim=rep_dim, hidden_dims=ae_hidden)

    def forward(self,x):
        rep = self.emb2rep(x)
        res_dict = self.ae(rep)
        res_dict['rep'] = rep
        return res_dict
