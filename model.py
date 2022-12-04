import torch
import torch.nn as nn
import yaml
from layer import STGRU
import pdb
from Normalize import Switch_Norm_2D

class model(nn.Module):
    def __init__(self,arg,A):
        super(model, self).__init__()
        self.A =A
        self.input_layer= torch.nn.Conv2d(arg.input_dim, arg.n_hid, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.stgru = STGRU(in_features= arg.num_nodes,gcn_in_channel=arg.n_hid, dp =arg.dropout)
        self.norm_1 = Switch_Norm_2D(arg.n_hid)
        self.output = nn.Sequential(nn.Linear(arg.n_hid, arg.n_hid) , nn.ReLU(),  nn.Linear(arg.n_hid,  arg.out_length) )
        self.dropout = nn.Dropout(p=arg.dropout)
        
    def forward(self, x): # x - > torch.Size([64, 12, 207, 2])
        x=x.permute(0,3,2,1)
        x = self.input_layer(x) 
        x= self.norm_1(x.permute(0,2,3,1))

        hidden = torch.ones_like(x[:, :, 1, :]).requires_grad_()
        #mask = torch.ones_like(self.A).requires_grad_()

        for i in range(x.size(2)):
            data = x[:,:, i, :]
            hidden  = self.stgru(data, hidden , self.A)
        
        out = hidden
        out = self.output(out)
        out= out.permute(0,2,1).unsqueeze(3)

        return out












# cl= Inception_Temporal_Layer(num_stations=207, In_channels=2, Hid_channels=4*2, Out_channels=4)
# inp =torch.rand(8,207,12,2)
# out = cl(inp)
# print(out.shape)


# A= torch.rand(207,207)
# x =torch.rand(64, 12, 207, 2)
# Model = mlstm(cfg['structure_params'],A)

# out = Model(x)
# print(out.shape)

# input_dim=3

# layer_z = nn.Conv2d(input_dim *2, input_dim*2, 1)
# layer_m= nn.Conv2d(input_dim *3, input_dim*3, 1)
# inp =torch.rand(8,3,224,224)
# W_z = torch.cat([inp,inp], dim = 1)
# Z = layer_z(W_z)
# combined = layer_m(torch.cat([Z, inp], dim = 1))
# mo, mg, mi = torch.split(combined, input_dim, dim = 1)
# print(mo.shape)


# import math

# def scaled_dot_product(q, k, v, mask=None):
#     pdb.set_trace()
#     d_k = q.size()[-1]
#     attn_logits = torch.matmul(q, k.transpose(-2, -1))
#     attn_logits = attn_logits / math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
#     attention = F.softmax(attn_logits, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention



# seq_len, d_k = 4, 207

# q = torch.randn(12,seq_len, d_k)
# k = torch.randn(12,seq_len, d_k)
# v = torch.randn(12,seq_len, d_k)
# values, attention = scaled_dot_product(q, k, v)
# print(values.shape)
# print("Q\n", q)
# print("K\n", k)
# print("V\n", v)
# print("Values\n", values)
# print("Attention\n", attention)

