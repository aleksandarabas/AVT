import requests
import pprint
import torch.utils.tensorboard as tb
import torch 
from requests.exceptions import HTTPError
import krakenex
import numpy as np
import csv
k = krakenex.API()
k.load_key('kraken.key')
import time

def krak_query(symb= 'XBTUSD'):
    l = k.query_public('Depth', {'pair': symb, 'count': '1'})
    r = l['result'][list(l['result'].keys())[0]]
    a= float(r['asks'][0][0])  
    a1 = float(r['asks'][0][1])
    b=float(r['bids'][0][0])
    b1 = float(r['bids'][0][1])
    return (a*a1+b*b1)/(a1+b1)

def kdata_gen(symb='XBTUSD', min_interval = 0.05):
    tlast= time.time()-min_interval
    while True:
        while time.time()-tlast< min_interval:
            time.sleep(0.01)
        tlast = time.time()
        yield krak_query(symb)



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

def wlistfromsymb(sym = ['XBTUSD'], size = 128, min_call_interval = 0.02):
    a = [kdata_gen(s, min_call_interval) for s in sym]
    wlists = [wlist(b, size = size) for b in a]
    while True:
        yret  = np.zeros((len(wlists), size))
        for i in range(len(wlists)):
            yret[i] = next(wlists[i])
        yield yret

omars = ['BTC', 'ETH', 'USD', 'EUR']

osmans = ['AAVE', 'ALGO', 'ANT', 'YFI', 'REPV2', 'BAL', 'BNT', 'XRP', 'ADA', 'LINK', 'COMP', 'ATOM', 'GRT', 'QTUM', 'MLN', 'OMG', 'MINA', 'FLOW', 'GNO', 'ICX', 'KAVA', 'KEEP', 'KSM']
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

river = []
for i in range(2):
    for j in osmans:
        river.append(omars[i]+j)
sz = 256
import torch.nn as nn
from basic_lstm import MMM

ma =MMM()
sz = 128
kernel_size = 16
num_convs = 3 

inps = wlistfromsymb(size=(sz*8 + 2*num_convs*(kernel_size+1) ), min_call_interval = 1)

ma.to(device)
ma.load_state_dict(torch.load('ckpt/mark7/45373.ckpt'))
ma.eval()
#optimizer = torch.optim.Adam(ma.parameters(), lr=1e-4)

fees = 1 - 0.24/100
h0, c0 = ma.init_states()
h0, c0 =  h0.to(device), c0.to(device)

hh0 = torch.randn(sz//4,1, sz//2).to(device)
cc0 = torch.randn(sz//4,1, sz//2).to(device)

balances = np.zeros([128, 2])
balances[:,0] = 40
v =0 

def tre_run(X, Y, h0, c0,test_lstm):
    output, (hn, cn) = test_lstm(X/(X[0]*2), (h0, c0))
    output = output*X[0]*2
    loss = loss_fn(output, Y)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    print( torch.sum(torch.abs((Y-X))))
    #print(output)
    #print(Y)
    h0 = hn.detach()
    c0 = cn.detach()
    return h0, c0
cc = 0
alpha = 0.1
sig = 0.5
qua =5
#bal = k.query_private('Balance')['result']
balances[0,0] = 40  #float(bal['ZUSD'])
balances[1,0] = 0  #float(bal['XXBT'])
while True:
    S = next(inps)[0].reshape(1,1,-1).astype(np.float32)
    #X = torch.from_numpy(S[:,:,:sz//2]).to(device)
    Y = torch.from_numpy(S).to(device)
    
    #X = torch.from_numpy(S).to(device)
    output, (hn, cn) = ma(Y, h0, c0)
    
    onp = output.cpu().data.numpy().squeeze()
    cntr = np.mean(onp)
    mmn = np.min(onp)
    mx = np.max(onp)
    #print(cntr, mmn, mx)
    h0 = hn.detach()
    c0 = cn.detach()
    cc+=1
    v = cntr*alpha + (1-alpha)*v
    print(cc)
    if cc > 300:
        #print("paper trading..")
        #print(bal)
        curp = 1

        t_thresh = 0.004
        print (np.max(onp), np.min(onp))
        print( onp[:10])
        print( S[0,0,-10:])
        if  np.max(onp)>=  t_thresh:
            balances[0, 1] = balances[0,1] + balances[0, 0]*qua* fees/ S[0,0,-1]
            
            #response = k.query_private('AddOrder',
            #                   {'pair': 'BTCUSD',
            #                     'type': 'buy',
            #                     'ordertype': 'market',
            #                     'volume': str(qua/ S[0,0,-1])},
            #                     )
            #bal = k.query_private('Balance')['result']
            balances[0,0] = float(bal['ZUSD'])
            balances[1,0] = float(bal['XXBT'])
            #pprint.pprint(response)
            #balances[0, 0] -= balances[0, 0]*qua
        else:
            if np.min(onp)<= -t_thresh:
                balances[0, 0] = balances[0, 0]+balances[0, 1]*qua* fees* S[0,0,-1]
                #response = k.query_private('AddOrder',
                #                    {'pair': 'BTCUSD',
                #                     'type': 'sell',
                #                     'ordertype': 'market',
                #                     'volume': str(qua/S[0,0,-1])},
                #                     )
                #bal = k.query_private('Balance')['result']
                balances[0,0] = float(bal['ZUSD'])
                balances[1,0] = float(bal['XXBT'])
                #pprint.pprint(response)
            #balances[0, 1]-=balances[0, 1] *qua
        print(f'{S[0,0,-1]*balances[1,0] + balances[0,0]}$')
        print(balances[0,0] , balances[1,0])
    #hh0, cc0= tre_run(X,Y,hh0,cc0, ma)
             