import torch
import torch.nn as nn
from torch import Tensor
import math,pdb
import torch.nn.functional as F

from modules import gcn_glu,GRUcell

class STGRU(nn.Module):

    def __init__(self,
                 in_features,
                 gcn_in_channel,
                 bias=True,
                 activation=F.relu,
                 dp=0.2):
        """GRUcell.
        :param in_features: Size of each input sample. In this case it is the number of nodes
        :param gcn_in_channel: dimension of the node feature of gcn.
        """
        super(STGRU, self).__init__()
        self.in_features = in_features
        self.activation = activation
        self.bias = bias
        self.act = nn.LeakyReLU()
        self.dp = nn.Dropout(p=dp)

        self.gcnglu = gcn_glu(gcn_in_channel, gcn_in_channel)
        #self.gcnglu2 = gcn_glu(gcn_in_channel, gcn_in_channel)
        self.lk = nn.LeakyReLU(0.1)

        #self.conv1d1 = nn.Conv1d(2*in_features, 2*in_features, kernel_size=1 )
        # self.conv1d2 = nn.Conv1d(3*in_features, 3*in_features, kernel_size=1 )
        # self.mlean1 = nn.Linear(gcn_in_channel,gcn_in_channel)
        # self.mlean2 = nn.Linear(gcn_in_channel,gcn_in_channel)

        self.ap =nn.AvgPool1d(kernel_size=2)

        self.linap =nn.Linear(32, gcn_in_channel)

        self.GRU = GRUcell(gcn_in_channel, gcn_in_channel)


    def forward(self, x,mem, A):  #torch.Size([64, 325, 4])

        zh =x
        x = self.lk(self.gcnglu(x.permute(1,0,2),A).permute(1,0,2)) 
        # x+=zh
        # x = self.lk(self.gcnglu(x.permute(1,0,2),A).permute(1,0,2)) 

        ## attentive attention
        C = torch.cat((mem,x),1)
        P=self.ap(C)
        P= self.linap(P)    
        SM= F.softmax(P, dim=1)
        a1,a2= torch.split(SM, self.in_features, dim = 1)
        a1 = a1*x
        a2= a2*mem
        zm = a1+a2
        y= torch.cat((x,zm),1)
        #pdb.set_trace()
        #y = self.conv1d1(y)
        y=torch.cat((y,zh),1)
        #y= self.conv1d2(y)
        wh, wr, wz = torch.split(y, self.in_features, dim = 1)

        mem = self.GRU(wh,wr,wz, mem)
        # mem = self.dp(mem)
        # mo, mg, mi = torch.split(y, self.in_features, dim = 1)

        # mi = torch.sigmoid(mi)
        # new_m = (1 - mi) * mem + mi * torch.tanh(mg)
        # new_h = torch.sigmoid(mo) * new_m 
        
        # return  new_h, new_m 

        return  mem
