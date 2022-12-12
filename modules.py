import torch
import torch.nn as nn
class GRUcell(nn.Module):
    def __init__(self,w_size, u_size):
        super(GRUcell, self).__init__()
        self.wh = torch.nn.Parameter(torch.Tensor(w_size,w_size))
        self.wr = torch.nn.Parameter(torch.Tensor(w_size,w_size))
        self.wz = torch.nn.Parameter(torch.Tensor(w_size,w_size))

        self.uz = torch.nn.Parameter(torch.Tensor(u_size,u_size))
        self.ur = torch.nn.Parameter(torch.Tensor(u_size,u_size))
        self.uh = torch.nn.Parameter(torch.Tensor(u_size,u_size))


        nn.init.xavier_uniform_(self.wh)
        nn.init.xavier_uniform_(self.wr)
        nn.init.xavier_uniform_(self.wz)

        nn.init.xavier_uniform_(self.uh)
        nn.init.xavier_uniform_(self.ur)
        nn.init.xavier_uniform_(self.uz)

    def forward(self, xh,xr,xz ,ht_1):

        z = torch.sigmoid( torch.matmul(xz,self.wz)+ torch.matmul(ht_1,self.uz))
        r = torch.sigmoid( torch.matmul(xr,self.wr)+ torch.matmul(ht_1,self.ur) )
        h_hat = torch.tanh( torch.matmul(xh, self.wh)+ r*torch.matmul(ht_1, self.uh))

        h= z*ht_1+(1-z)*h_hat

        return h


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self, A, x):
        x = torch.einsum('vn,bfnt->bfvt',(A,x))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self,x):
        return self.mlp(x)

class gcn_glu(nn.Module):
    def __init__(self,c_in,c_out):
        super(gcn_glu,self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,2*c_out)
        self.c_out = c_out
    def forward(self, x, A):
        
        # (N, B, C)
        x = x.unsqueeze(3) # (N, B, C, 1)
        x = x.permute(1, 2, 0, 3) # (N, B, C, 1)->(B, C, N, 1)
        ax = self.nconv(A,x)
        axw = self.mlp(ax) # (B, 2C', N, 1)
        axw_1,axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)
        axw_new = axw_1 * torch.sigmoid(axw_2) # (B, C', N, 1)
        axw_new = axw_new.squeeze(3) # (B, C', N)
        axw_new = axw_new.permute(2, 0, 1) # (N, B, C')
        return axw_new

