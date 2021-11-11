import requests
import pprint
import torch.utils.tensorboard as tb
import torch 
from requests.exceptions import HTTPError
import krakenex

URL = 'https://www.alphavantage.co/query?'
def GdatK(params, URL=URL):
    params['apikey']= 'KIYIMT6RHH43I5MO'
    
    with requests.Session() as s:
        r = requests.get(URL, params=params)
        dc = r.json()
    return dc[list(dc.keys())[1]]

with open("all_tickers.txt","r") as f:
    c = f.read()
call = c.split('\n')


#r= GdatK(params={'function': 'TIME_SERIES_INTRADAY', 'symbol':'IBM','interval': '5min'})
#print(r)




k = krakenex.API()
k.load_key('kraken.key')

#response = k.query_public('Depth', {'pair': 'BTCUSD', 'count': '1'})

#response = k.query_private('AddOrder',
#                                {'pair': 'ETHUSD',
#                                 'type': 'buy',
#                                 'ordertype': 'market',
#                                 'volume': '0.004'}
#                                 )
import numpy as np
import csv
def kdata_gen(symb='XBTUSD', startdate=1586000205):
    with open(f'kdata/{symb}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if int(row[0])<startdate:
                continue
            startdate= int(row[0])+1
            yield(row)
def  itfl(genl, idx = 1):
    while True:
        r = next(genl)
        if r is None:
            return None
        yield float(r[idx])

def wlist(genl, size = 128):
    l = np.zeros(size)
    for i in range(size):
        l[i] = next(genl)
    while True:
        nx = next(genl)
        if nx==None:
            return None
        l = np.roll(l,-1)
        l[-1] = nx
        yield l

def wlistfromsymb(sym = 'XBTEUR', size = 128):
    a = kdata_gen(sym)
    b = itfl(a)
    return wlist(b, size = size)
sz = 128
kernel_size = 16
num_convs = 3 
DSXBTUSD = wlistfromsymb(size=(sz*9 + 2*num_convs*(kernel_size+1) ))
szz = sz*9 + 2*num_convs*(kernel_size+1) 
#DSXBTETH = wlistfromsymb('XBTETH')
linps = [wlistfromsymb('ETHUSD',size=szz), wlistfromsymb('XRPEUR',size=szz), wlistfromsymb('LTCEUR',size=szz), wlistfromsymb('LTCUSD',size=szz),  wlistfromsymb('XRPUSD',size=szz), wlistfromsymb('EURUSD',size=szz), wlistfromsymb('XLMEUR',size=szz)]
import torch.nn as nn
from basic_lstm import MMM
test_lstm =MMM()

device = "cuda" if torch.cuda.is_available() else "cpu"
test_lstm.load_state_dict(torch.load('ckpt/mark7/45373.ckpt'))
test_lstm = test_lstm.to(device)

def noise_w(lstms):
    for m in lstms:
        for w in m.all_weights:
            for ww in w:
                ww.data *= torch.normal(mean = 1e-10, std = 1e-9, size = ww.shape).to(device)
#noise_w(test_lstm)
loss_fn = nn.L1Loss()
optimizer1 = torch.optim.RMSprop(test_lstm.parameters(), lr=1e-8)
optimizer2  = torch.optim.Rprop(test_lstm.head.parameters(), lr = 1e-8)
import os
import time
import datetime
tdir =  "ckpt/" + 'mark8'
try:
    os.mkdir(tdir)
except:
    print("yee")

lt = []
cc= 0
notr = 3
lloss  = 0
dlloss = 0
adlr = 1
notrh=3
fl =0
dprob = 0.1

lstsave = time.time()
skip_time = [100, 5000]
skip_probability = 0.1
h0, c0 = test_lstm.init_states()
h0, c0 =  h0.to(device), c0.to(device)
lstates =[]
for i in range(len(linps)):
    h0, c0 = test_lstm.init_states()
    h0, c0 =  h0.to(device), c0.to(device)
    lstates.append([h0,c0])
@torch.jit.script
def loss_fn2(G, Y):
    
    dG = G[1:] - G[:-1]
    dY = Y[1:] - Y[:-1]

    return torch.sum(torch.abs(dY-dG))
@torch.jit.script
def loss_fn3(G, Y):
    iG = torch.zeros_like(G)
    iY = torch.zeros_like(Y)
    iG[0]= G[0]
    iY[0]= Y[0]
    for i in range(1, iG.shape[0]):
        iG[i] = iG[i-1] + G[i]
        iY[i] = iY[i-1] + Y[i]
    return  torch.sqrt(torch.sum((iY-iG)**2))
@torch.jit.script
def loss_fn4(G, Y):
    iG = torch.zeros_like(G)
    iY = torch.zeros_like(Y)
    iG[0]= G[0]
    iY[0]= Y[0]
    for i in range(1, iG.shape[0]):
        iG[i] = iG[i-1] + torch.abs(G[i])
        iY[i] = iY[i-1] + torch.abs(Y[i])
    return  torch.sqrt(torch.sum((iY-iG)**2))
@torch.jit.script
def loss_fn5(G, Y):
    tot = torch.zeros_like(G[0])
    for i in range(G.shape[0]-1):
        tot+= ((1.0)/(G.shape[0]))*torch.tan(G[i]-G[i+1])/torch.tan(Y[i]-Y[i+1])
    return tot
losses = [loss_fn, loss_fn2, loss_fn3, loss_fn4] 

iloss_inrow = 0
lst_loss = 100000
while True:
    
    loss =0
    notrh = max(notrh, 3)
    notr = np.random.randint(1, high = notrh)

    cur_i = cc%len(linps)
    
    for i in range(notr):
        S = next(linps[cur_i]).reshape(1,1,-1).astype(np.float32)
        
        X = torch.from_numpy(S[:,:,:-sz]).to(device)
        Y = torch.from_numpy(S[:,:,-sz:]).to(device)
        Y = (Y-X.view(-1)[-1])/X.view(-1)[-1]   
        #print(X.shape, Y.shape)
        output, (hn, cn) = test_lstm(X, lstates[cur_i][0], lstates[cur_i][1])
        h0 = hn.detach()*torch.normal(mean = 1e-10, std = 1e-9, size = hn.shape).to(device)
        c0 = cn.detach()*torch.normal(mean = 1e-10, std = 1e-9, size = cn.shape).to(device)
        lstates[cur_i] = [h0,c0]
        #if np.random.normal(loc = 0.5) < dprob:
        #    loss += loss_fn2(output, Y)
    # Backpropagation
    
    loss = losses[cc%len(losses)](output.view(-1), Y.view(-1))  
    if cc%100<50:
        optimizer = optimizer1
    else:
        optimizer = optimizer2
    optimizer.zero_grad()
    #optimizer2.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    #optimizer2.step()

    if cc% 100 ==0:
        print(f'iteration #{cc}, Loss: {loss.data}, notr: {notr}')
        print(f'GT{Y.data[:,:,:5]}')
        print(f'OT{output.data[:,:,:5]}')
    #print(output)
    #print(Y)
    #h0 = hn
    #c0 = cn
    #lt.append([hn,cn])
  
    cc+=1

    if time.time()- lstsave > 60:
        torch.save(test_lstm.state_dict(), os.path.join(str(tdir), f'{cc}.ckpt'))
        lstsave = time.time()
        
        notrh+=1
    
    notrh%=200
    if lst_loss>loss:
        iloss_inrow+=1
    else:
        iloss_inrow=0
    lst_loss = loss
    if iloss_inrow > 2:
       noise_w(test_lstm.BLSTM.lstms)
       iloss_inrow=0
