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
    print(symb)
    print("")
    print("")
    r = l['result'][list(l['result'].keys())[0]]
    a= float(r['asks'][0][0])  
    a1 = float(r['asks'][0][1])
    b=float(r['bids'][0][0])
    b1 = float(r['bids'][0][1])
    return (a*a1+b*b1)/(a1+b1)

def kdata_gen(symb='XBTUSD', min_interval = 0.05):
    tlast= 0
    while True:
        if time.time()-tlast< min_interval:
            yield -1
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

def wlistfromsymb(sym = ['XBTUSD'], size = 128, min_call_interval = 1):
    a = [kdata_gen(s, min_call_interval) for s in sym]
    wlists = [wlist(b, size = size) for b in a]
    lst_call = 0
    while True:
        if time.time() - lst_call < min_call_interval:
            time.sleep(lst_call + min_time_interval - time.time())
        lst_call = time.time()
        yret  = np.zeros((len(wlists), size))
        for i in range(len(wlists)):
            yret[i] = next(wlists[i])
        yield yret

omars = [ 'USD', 'EUR']

osmans = ['BTC', 'ETH','ALGO', 'ANT', 'YFI', 'REPV2', 'BAL', 'BNT', 'XRP', 'ADA', 'LINK', 'COMP', 'ATOM', 'GRT', 'QTUM', 'MLN', 'OMG', 'MINA', 'FLOW', 'GNO', 'ICX', 'KAVA', 'KEEP', 'KSM']
device = "cuda" if torch.cuda.is_available() else "cpu"

river = []
for i in range(2):
    for j in osmans:
        river.append(j+omars[i])

inps = wlistfromsymb(sym = river, size = 1, min_call_interval = 1)

filelist = [open('data_live_collected/'+r+'.txt', 'w') for r in river]

while True:
    inp = next(inps)
    for i in range(len(filelist)):
        filelist[i].write(str(inp[i][0]) + '\n')
 