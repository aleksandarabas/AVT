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
def kdata_gen(symb='XBTUSD', startdate=1489966205):
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

def wlistfromsymb(sym = 'XBTUSD', size = 128):
    a = kdata_gen(sym)
    b = itfl(a)
    return wlist(b, size = size)
sz = 128
kernel_size = 16
num_convs = 3 
DSXBTUSD = wlistfromsymb(size=(sz*9 + 2*num_convs*(kernel_size+1) ))
#DSXBTETH = wlistfromsymb('XBTETH')
#DSETHUSD = wlistfromsymb('USD')
import torch.nn as nn
from basic_lstm import MMM
test_lstm =MMM()

device = "cuda" if torch.cuda.is_available() else "cpu"
test_lstm.load_state_dict(torch.load('ckpt/mark1/30502.ckpt'))
test_lstm = test_lstm.to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.RMSprop(test_lstm.parameters(), lr=1e-4)
#optimizer2  = torch.optim.Adam(test_lstm.head.parameters())
import os
import time
import datetime
tdir =  "ckpt/" + 'mark2'
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
while True:
    loss =0
    notr = np.random.randint(1, high = notrh)

    if np.random.normal(loc = 0.5) <skip_probability:
        h0, c0 = test_lstm.init_states()
        h0, c0 =  h0.to(device), c0.to(device)
        skip_num = np.random.randint(skip_time[0], skip_time[1])
        for i in range(skip_num):
            next(DSXBTUSD)
    for i in range(notr):
        S = next(DSXBTUSD).reshape(1,1,-1).astype(np.float32)
        
        X = torch.from_numpy(S[:,:,:-sz]).to(device)
        Y = torch.from_numpy(S[:,:,-sz:]).to(device)
        Y = 100000*(Y-X.view(-1)[-1])/X.view(-1)[-1]   
        #print(X.shape, Y.shape)
        output, (hn, cn) = test_lstm(X, h0, c0)
        h0 = hn.detach()
        c0 = cn.detach()
        output = output*100000
        if np.random.normal(loc = 0.5) < dprob:
            loss += loss_fn(output, Y)
    # Backpropagation
    loss += loss_fn(output, Y)
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
    
    notrh%=5000
