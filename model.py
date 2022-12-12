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


