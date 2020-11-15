#!/usr/bin/env python
# coding: utf-8

# In[1]:


strategy_name ='培宏量化1号'


# In[2]:


import sys
sys.path.append("C:\Program Files\Tinysoft\Analyse.NET")
sys.path.append(r"C:\Users\xudong\Documents\github\coresearch\funcs")
import rschLib
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import datetime
import copy
import pandas as pd
from operator import itemgetter
import TSLPy3 as tsl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import talib
import matplotlib.dates as mdates

np.set_printoptions(formatter={'float_kind': "{:.6f}".format})
client = pymongo.MongoClient('localhost', 27017)
db = client.quanLiang
dbt = client.tinySoftData


# In[3]:


def nowTime():
    return int(time.strftime("%H%M%S",time.localtime(time.time())))
def nowString():
    return time.strftime("%Y%m%d %H:%M:%S",time.localtime(time.time()))
nowTime()
nowString()


# In[4]:


with open(r"d:\pkl\dailyBarMtx.pkl", 'rb+') as f:
    z = pickle.load(f)
dtes = z['dtes']
dtes = np.array(dtes).astype(np.int)
tkrs = list(z['tkrs'])
open_mtx = z['open_mtx']
high_mtx = z['high_mtx']
low_mtx = z['low_mtx']
close_mtx = z['close_mtx']

k = close_mtx==0
close_mtx = rschLib.zero2pre(close_mtx)
open_mtx[k]=close_mtx[k]
high_mtx[k]=close_mtx[k]
low_mtx[k]=close_mtx[k]

name = z['name']
belong = z['belong']
shenwan1 = z['shenwan1']
shenwan2 = z['shenwan2']
shenwan3 = z['shenwan3']
vol_mtx = z['vol_mtx']
amount_mtx = z['amount_mtx']


# In[5]:


v = pd.DataFrame(vol_mtx)
q = np.array(v.rolling(5,axis=1).mean())
q = np.array(q)
lb=vol_mtx[:, -1]/q[:,-2]
lb[np.isfinite(lb)==False]=0


# In[7]:


Wl = 500 # 当天收盘价格位于Wl日内的高低点相对位置
preHighL = np.max(high_mtx[:, -Wl:-1], axis=1)
preLowL = np.min(low_mtx[:, -Wl:-1], axis=1)
priceLocL = (close_mtx[:,-2]-preLowL)/(preHighL-preLowL)
# 1. 价格位置
idxPriceLoc = priceLocL < 0.5
namePriceLoc = name[idxPriceLoc]
# 2. 
idxTiaoKongGaoKai = ((open_mtx[:,-1] / high_mtx[:,-2]) - 1) > 0.01
nameTiaoKongGaoKai = name[idxTiaoKongGaoKai]
# 3.
nameMarketValue = [x['name'] for x in list(db.tkrsInfo.find({'tagCirculateMarketValueBiggerThan100Y':1}, {'name':1}))]
# 4. 当天是上涨的
nameIsUp = name[close_mtx[:, -1]>open_mtx[:, -1]]


# In[8]:


m = set(namePriceLoc).intersection(set(nameMarketValue)).intersection(set(nameTiaoKongGaoKai)).intersection(set(nameIsUp))
lm = [tkrs[list(name).index(x)] for x in m]
qt = list(dbt.minuteBarStock.find({'ticker':{'$in': list(lm)},'sale1':{'$gt':0},'dateAsInt':int(dtes[-1]),"offSetFromMidNight" :  5370e4}, {'ticker':1, 'close':1, 'open':1, 'sectional_open':1, 'lb':1, 'sale1':1, 'dateTime':1, 'StockName':1}).sort('lb',-1)) 
selectedName=[x['StockName'] for x in qt]


# In[9]:


if (len(selectedName)>0):
    s='可交易标的:'
    for x in selectedName:
        s = s + x + ' '
else:
    s='没有符合条件标的'
s = s+' ('+str(dtes[-1])+')'
print(s)
db.strategyEventRecords.insert_one({'strategy_name':strategy_name, 'updateTime':nowString(), 'content':s})
      

