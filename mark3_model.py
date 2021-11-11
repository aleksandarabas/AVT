import torch
from torch import nn
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

    def forward(self, X, h0, c0):
        
        for i in range(self.num_iter):
            o, (a,b)=  self.lstms[0](self.zoom*X, (h0, c0))
            hn=  a
            cn=  b
            for lstm in self.lstms[1:]:
                o, (a,b)=   lstm(o, (hn, cn))
                hn= (a )
                cn= (b )

        return o, (hn, cn)

    def init_states(self):
        return torch.randn(self.num_layers,1, self.sz),torch.randn(self.num_layers,1, self.sz)


def makeconv(channels_in = 1, channels_out =1, kernel_size = 16, stride =2):
    conv = nn.Conv1d(channels_in, channels_out, kernel_size, stride)
    bn = nn.BatchNorm1d(num_features =1)
    act = nn.ReLU()
    return nn.Sequential(*list([conv, bn, act]))

class hHead(nn.Module):
    def __init__(self, 
                 num_convs  = 3,
                 ):
        super(hHead, self).__init__()
    
        self.convs1d = nn.Sequential( * list( makeconv() for i in range(num_convs)))

    def forward(self, X):
        return self.convs1d(X)



class MMM(nn.Module):
    def __init__(self,
                 num_convs = 3,
                 sz = 128,
                 zoom = 1,
                 num_layers = 4,
                 num_lstms = 8, 
                 num_iter =4,
                 ):
        super(MMM, self).__init__()

        self.head = hHead(num_convs = num_convs)
        self.BLSTM = BasicLSTM(sz = sz,
                 zoom = zoom,
                 num_layers = num_layers,
                 num_lstms = num_lstms, 
                 num_iter =num_iter)
    
    def forward(self, X, h0, c0):
        X1 = (X-X.view(-1)[-1])/(X.view(-1)[-1])
        x_i = self.head(X1)
        return self.BLSTM(x_i, h0, c0)
    
    def init_states(self):
        return self.BLSTM.init_states()
