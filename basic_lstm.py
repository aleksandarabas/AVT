import torch
from torch import nn
import numpy as np
class BasicLSTM(nn.Module):
    def __init__(self, 
                 sz = 128,
                 zoom = 1,
                 num_layers = 4,
                 num_lstms = 8, 
                 num_iter =3,
                 ):
        super(BasicLSTM, self).__init__()

        self.lstms = nn.Sequential(*list([ 
                   nn.LSTM(input_size = sz,
                   hidden_size= sz,
                   num_layers= num_layers, dropout=0.2) for i in range(num_lstms)]) )
        self.zoom = zoom
        self.sz = sz
        self.num_layers = num_layers
        self.num_iter = num_iter

    def forward(self, X, h0, c0, lp):
        s = X
        hn = h0
        cn = c0
        r = torch.zeros_like(X)
        
        for i in range(self.num_iter):
            o, (a,b)=  self.lstms[0](torch.tanh((X+s)/2), (hn, cn))
            s =  f(s+o)
            hn=  a
            cn=  b
            ll=0
            for lstm in self.lstms[1:]:    
                o, (a,b)=   lstm(torch.tanh((s+X)/2), (hn, cn))
                if ll%2==0:
                    mul=-1
                else:
                    mul=1
                r= r + mul*torch.relu(o)
                s = f(s+o)
                hn= b
                cn= a       
                ll+=1
                
                 
        #o=f(o)
        return r, (hn, cn)
    
    @torch.jit.script
    def p(o):
        l= o>(1e+2)
        z=torch.zeros_like(o)
        z[l] =1

        return o*(1-z) + o*z*(-1e-3)
    
    def init_states(self):
        return torch.zeros(self.num_layers,1, self.sz),torch.zeros(self.num_layers,1, self.sz)
@torch.jit.script
def f(o):
    l= o<(-1e-11)
    z=torch.zeros_like(o)
    z[l] =1

    return o*(1-z) + o*z*(-1e+10)




def makeconv(channels_in = 1, channels_out =1, kernel_size = 16, stride =2):
    conv = nn.Conv1d(channels_in, channels_out, kernel_size, stride)
    bn = nn.BatchNorm1d(num_features =1)
    act = nn.Tanh()
    return nn.Sequential(*list([conv, bn, act]))

class hHead(nn.Module):
    def __init__(self, 
                 num_convs  = 3,
                 ):
        super(hHead, self).__init__()
    
        self.convs1d = nn.Sequential( * list( makeconv() for i in range(num_convs)))

    def forward(self, X):
        for conv in self.convs1d:
            X = conv(X)
            X = f(X)
        return X



class MMM(nn.Module):
    def __init__(self,
                 num_convs = 3,
                 sz = 128,
                 zoom = 1,
                 num_layers = 4,
                 num_lstms = 8, 
                 num_iter =3,
                 ):
        super(MMM, self).__init__()
        self.sz =sz
        self.head = hHead(num_convs = num_convs)
        self.BLSTM = BasicLSTM(sz = sz,
                 zoom = zoom,
                 num_layers = num_layers,
                 num_lstms = num_lstms, 
                 num_iter =num_iter)
    
    def forward(self, X, h0, c0):
        X1 = (X-X.view(-1)[-1])/(X.view(-1)[-1])
        x_i = self.head(X1)
        
        return self.BLSTM(x_i, h0, c0, torch.log1p(torch.max(torch.abs(X1).view(-1))))
    
    def init_states(self):
        return self.BLSTM.init_states()
