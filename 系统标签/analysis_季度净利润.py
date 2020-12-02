#!/usr/bin/env python
# coding: utf-8

# In[301]:


strategy_name = 'tagNetProfitGrowth3YBiggerThan20P'
offStart = ('close_mtx',0)


# In[303]:


import sys
sys.path.append("C:\Program Files\Tinysoft\Analyse.NET")
sys.path.append(r"C:\Users\xudong\Documents\github\coresearch\funcs")
import pymongo
import numpy as np
import pickle
import time
import datetime
import copy
import pandas as pd
from operator import itemgetter
import TSLPy3 as tsl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import talib
import importlib
import rschLib
np.set_printoptions(formatter={'float_kind': "{:.6f}".format})
client = pymongo.MongoClient('localhost', 27017)
db = client.quanLiang
dbt = client.tinySoftData
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx()


# In[304]:


maxD = 5
inTime = 234
otTime = 474
tradeArea=[inTime,otTime]
timeAsFloat, timeLabels, maxM, dayOff, dayTimeAsFloat = rschLib.getTimeLabels(maxD)
importlib.reload(rschLib)
tradesUsed, Po, r, Sale1 = rschLib.getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat)


# In[124]:


R = open_mtx[:, 1:]/close_mtx[:,:-1]-1
R = np.hstack((np.zeros((R.shape[0],1)), R))


# In[210]:


dicttkrs = dict(zip(tkrs, range(len(tkrs))))
dictdtes = dict(zip(dtes, range(len(dtes))))


# In[187]:


q = np.array([int(x) for x in np.floor(dayTimeAsFloat)])
dtesUsed = [x['dateIn'] for x in tradesUsed]
idxOpen = np.nonzero(np.round(dayTimeAsFloat-np.floor(dayTimeAsFloat), 4)==0.0931)[0]
for i in range(r.shape[0]):
    keys = 


# In[297]:


timeLabels[240]
timeLabels[240]


# In[270]:


i = len(tradesUsed)-251
lendtes = len(dtes)
k1 = dicttkrs[tradesUsed[i]['ticker']]
k2 = dictdtes[tradesUsed[i]['dateIn']]
maxl = np.min((k2+maxD, len(dtes)))
print(r[i, idxOpen[:maxl-k2]], R[k1, k2:maxl])
#r[i, idxOpen[:maxl-k2]] = R[k1, k2:maxl]


# In[298]:


Po[i, idxOpen]
Po[i, idxOpen-1]


# In[300]:


timeLabels[239]


# In[ ]:


问题，缺失最后一秒的开盘到收盘的回报率，以至于无法计算正确的隔夜回报率。 
解决方案：1. r中的每天第一分钟，应该是隔夜收盘


# In[281]:


Po[i, 230:240]


# In[286]:


timeLabels[230:241]


# In[271]:


tradesUsed[-251]


# In[273]:


15.18/15.15-1


# In[209]:


dictdtes


# In[207]:


R.shape


# In[167]:


i = 0
j = 0
dictOpenFromCloseByDtesAndTkrs = {}
strdtes = [str(x) for x in dtes]
a = datetime.datetime.now()
for i in range(len(dtes)-750,len(dtes)):
    for j in range(len(tkrs)):
        dictOpenFromCloseByDtesAndTkrs[strdtes[i]+tkrs[j]]=R[j,i]
b = datetime.datetime.now()
print(b-a)


# In[120]:


Po = rschLib.zero2pre(Po)
P = Po
r=P[:,1:]/P[:,:-1] - 1
r=np.hstack((np.zeros((r.shape[0],1)),r))


# In[ ]:





# In[122]:


m = np.min(r, axis=1)
idx = np.nonzero(m<-0.2)[0]
dd = np.array(dayTimeAsFloat)
for j in idx:
    pp = r[j,:]<-0.2
    if (np.round((dd[pp]-np.floor(dd[pp]))[0],4)==0.0931):
        r[j,pp] = 0
    else:
        print(np.min(r[j,:]), (dd[pp]-np.floor(dd[pp])), dd[pp])


# In[123]:


[a,b]=np.min


# In[114]:


pp = r[j,:]<-0.1
if (np.round((dd[pp]-np.floor(dd[pp]))[0],4)==0.0931):
    r[j,pp] = 0
else:
    print(np.min(r[j,:]), (dd[pp]-np.floor(dd[pp])), dd[pp])


# In[115]:


r[j, pp]


# In[116]:


np.nonzero(r==np.min(r))


# In[109]:


j = 14249
pp = r[j,:]<-0.1
np.round((dd[pp]-np.floor(dd[pp]))[0],4)==0.0931
r[j,pp]=0
r[j,pp]


# In[39]:


plt.plot((Po[56,:]))


# In[25]:


np.quantile(r,0.00001)


# In[4]:


importlib.reload(rschLib)
trades, tradesUsed, Po, r = rschLib.getTrades(strategy_name, name, tkrs, dtes, maxD, maxM)


# In[ ]:


importlib.reload(rschLib)
result = rschLib.getTradeAnalysisSampleGroups(r, tradeArea)


# In[ ]:


#h = np.max(np.cumsum(r[:, :tradeArea[0]], axis=1), axis=1)
#isZhangtingBeforeTradeArea = h>=0.05
importlib.reload(rschLib)
rschLib.drawPriceChange(r, strategy_name, timeLabels=timeLabels, tp=tradeArea)
rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood20'], strategy_name, timeLabels=timeLabels, title='盈利前20%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前20%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易', tp=tradeArea)


# In[ ]:


importlib.reload(rschLib)
rschLib.analyzeTradeTags(trades, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(trades, result['rGood20'], result['idxGood20'], '盈利前20%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(trades, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(trades, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(trades, result['rBad20'], result['idxBad20'], '亏损前20%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(trades, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, name, offStart)


# In[ ]:


importlib.reload(rschLib)
[dtesPnl,pnl, numTrades]=rschLib.getPnl(dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes')


# In[ ]:


importlib.reload(rschLib)
rschLib.pnlVsNumtrades(pnl, numTrades, strategy_name)


# In[ ]:





# In[ ]:





# In[ ]:




