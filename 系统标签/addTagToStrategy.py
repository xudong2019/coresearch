#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# get time labels
# get trades


# In[3]:


q = db.taskAddTagToStrategy.find_one()
qm = db.strategyMinuteBar.find_one({'strategy_name':q['strategy_name_original']})
inTime, otTime = qm['concernPoints']


# In[7]:


maxD = 5
timeAsFloat, timeLabels, maxM, dayOff = rschLib.getTimeLabels(maxD)
importlib.reload(rschLib)
tt, tradesUsed, Po, r = rschLib.getTradesWithPklCache(q['strategy_name_original'], name, tkrs, dtes, maxD, maxM)


# In[9]:


m = db.strategyBackTest.find_one({'strategy_name':q['strategy_name_original']})
if ('off_start' in m.keys()):
    strategy_off_start = (m['off_start'][0], m['off_start'][1])


# In[11]:


d = rschLib.tagDict()
k = list(d.keys())
t = list([x['file'] for x in d.values()])
t.index(q['tagToAdd'])
tagName = k[t.index(q['tagToAdd'])]
tag_off_start = d[tagName]['off_start']
with open("d:\\pkl\\" + q['tagToAdd'] + ".pkl", 'rb+') as f:
    tagFile = pickle.load(f)
    tagMtx = tagFile['tag_mtx']    


# In[12]:


[p,idxTradesOverLapped] = rschLib.totInTag(range(len(tradesUsed)), tagMtx, dtes, name, tradesUsed, strategy_off_start, tag_off_start)


# In[13]:


len(idxTradesOverLapped)


# In[14]:


trades = [tradesUsed[x] for x in idxTradesOverLapped]


# In[15]:


tradesUsed = trades


# In[16]:


Po = Po[idxTradesOverLapped, :]


# In[17]:


r = r[idxTradesOverLapped, :]


# In[18]:


tradeArea=[inTime,otTime]
result = rschLib.getTradeAnalysisSampleGroups(r, tradeArea)


# In[19]:


strategy_name = q['strategy_name']
offStart = strategy_off_start


# In[20]:


rschLib.drawPriceChange(r, strategy_name, timeLabels=timeLabels, tp=tradeArea)


# In[22]:


importlib.reload(rschLib)
rschLib.drawPriceChange(r, strategy_name, timeLabels=timeLabels, tp=tradeArea)
rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood20'], strategy_name, timeLabels=timeLabels, title='盈利前20%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad20'], strategy_name, timeLabels=timeLabels, title='亏损前20%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad30'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易', tp=tradeArea)


# In[145]:


rschLib.analyzeTradeTags(tradesUsed, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rGood20'], result['idxGood20'], '盈利前20%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rBad20'], result['idxBad20'], '亏损前20%交易',strategy_name, dtes, name, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, name, offStart)


# In[23]:


#get tag names
importlib.reload(rschLib)
tnames, idxOverLapTagList,tagNamesEn=rschLib.analyzeTradeTags(tradesUsed, r, list(range(len(tradesUsed))), '所有交易',strategy_name, dtes, name, offStart)


# In[24]:


len(tradesUsed)


# In[25]:


#draw pnl and tag pnl
importlib.reload(rschLib)
[dtesByTrade, pnlByTrade]=rschLib.getPnl(dtes,tkrs, name, tradesUsed, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
[dtesPnlAggr,pnlAggr, numTrades] = rschLib.aggregatePnlAndDtes(dtesByTrade, pnlByTrade)
rschLib.drawPNL(dtesPnlAggr, pnlAggr, dtes, strategy_name, toDatabase='yes')


# In[26]:


#draw pnl and tag pnl
importlib.reload(rschLib)
for i in range(len(tnames)):
    tagName = tnames[i]
    [dtesWithTag, pnlWithTag,n] = rschLib.aggregatePnlAndDtes(dtesByTrade[idxOverLapTagList[i]],pnlByTrade[idxOverLapTagList[i]])
    rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name, toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType=tagName)
    rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name+'+'+tagNamesEn[i], toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType='pnl')

#analysis of number of trades vs performance
importlib.reload(rschLib)
rschLib.pnlVsNumtrades(pnlAggr, numTrades, strategy_name, toDatabase='yes')


# In[31]:


db.taskAddTagToStrategy.delete_one({'_id':q['_id']})

