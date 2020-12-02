#!/usr/bin/env python
# coding: utf-8

# In[1]:


maxD = 5
inTime = 240
otTime = 480


# In[2]:


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
#db = rschLib.db_quanLiang()
#dbt = rschLib.db_tinySoftData()
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx("D:\\pklWeeklyUpdate\\")
# get time labels
# get trades


# In[13]:


strategy_names = [x['file'] for x in rschLib.tagDict().values()]
offStarts = [x['off_start'] for x in rschLib.tagDict().values()]
importlib.reload(rschLib)


# In[73]:


def analyzeStrategy(strategy_name, offStart, dtes, name, tkrs):
    timeAsFloat, timeLabels, maxM, dayOff, dayTimeAsFloat = rschLib.getTimeLabels(maxD)
    R = open_mtx[:, 1:]/close_mtx[:,:-1]-1  #使用收盘到开盘的回报率来修正分红和拆股
    R = np.hstack((np.zeros((R.shape[0],1)), R))
    tradesUsed, r_withnan = rschLib.getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat, R)
    # get trade samples by good/bad trades
    tradeArea=[inTime,otTime]
    idxTradable = np.isfinite(r_withnan[:,tradeArea[0]])
    r = r_withnan.copy()
    r[np.isfinite(r)==False]=0
    result = rschLib.getTradeAnalysisSampleGroups(r, idxTradable, tradeArea)

    # draw price change
    rschLib.drawPriceChange(r[idxTradable,:], strategy_name, timeLabels=timeLabels, tp=tradeArea)
    rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
    #rschLib.drawPriceChange(result['rGood20'], strategy_name, timeLabels=timeLabels, title='盈利前20%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易',  tp=tradeArea)
    #rschLib.drawPriceChange(result['rBad20'], strategy_name, timeLabels=timeLabels, title='亏损前20%交易',  tp=tradeArea)
    rschLib.drawPriceChange(result['rBad30'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易',  tp=tradeArea)
    
    # analyze tags
    #rschLib.analyzeTradeTags(tradesUsed, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, tkrs, offStart)
    #rschLib.analyzeTradeTags(tradesUsed, result['rGood20'], result['idxGood20'], '盈利前20%交易',strategy_name, dtes, tkrs, offStart)
    #rschLib.analyzeTradeTags(tradesUsed, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, tkrs, offStart)
    #rschLib.analyzeTradeTags(tradesUsed, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, tkrs, offStart)
    #rschLib.analyzeTradeTags(tradesUsed, result['rBad20'], result['idxBad20'], '亏损前20%交易',strategy_name, dtes, tkrs, offStart)
    #rschLib.analyzeTradeTags(tradesUsed, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, tkrs, offStart)

    #get tag names
    tnames, tagNamesEn,t2 = rschLib.getTagNames()
    idxOverLapTagList=rschLib.analyzeTradeTags(tradesUsed, r, list(range(len(tradesUsed))), '所有交易',strategy_name, dtes, tkrs, offStart, "d:\\pklWeeklyUpdate\\")

    #draw pnl and tag pnl
    importlib.reload(rschLib)
    [dtesByTrade, pnlByTrade] = rschLib.getPnlFast(r, dtes, tkrs, name, tradesUsed, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
    [dtesPnlAggr,pnlAggr, numTrades] = rschLib.aggregatePnlAndDtes(dtesByTrade, pnlByTrade)
    rschLib.drawPNL(dtesPnlAggr, pnlAggr, dtes, strategy_name,showFigure='no', toDatabase='yes')
    for i in range(len(tnames)):
        tagName = tnames[i]
        [dtesWithTag, pnlWithTag,n] = rschLib.aggregatePnlAndDtes(dtesByTrade[idxOverLapTagList[i]],pnlByTrade[idxOverLapTagList[i]])
        rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name, showFigure='no', toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType=tagName)
        rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name+'+'+tagNamesEn[i], showFigure='no',  toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType='pnl')

    #analysis of number of trades vs performance
    importlib.reload(rschLib)
    rschLib.pnlVsNumtrades(pnlAggr, numTrades, strategy_name, toDatabase='yes')
    rschLib.saveOffStart(strategy_name, offStart)


# In[ ]:


import threadpool
importlib.reload(rschLib)
arg_list=[]#存放任务列表  
#首先构造任务列表  
for (i,strategy_name) in enumerate(strategy_names):
    if i<8:
        continue
    offStart = offStarts[i]
    analyzeStrategy(strategy_name, offStart, dtes, name, tkrs)


# In[ ]:


import threadpool
importlib.reload(rschLib)
arg_list=[]#存放任务列表  
#首先构造任务列表  
for (i,strategy_name) in enumerate(strategy_names):
    offStart = offStarts[i]
    arg_list.append(([],{'strategy_name':strategy_name, 'offStart':offStart, 'dtes':dtes, 'name':name, 'tkrs':tkrs}))
pool = threadpool.ThreadPool(1) 
requests = threadpool.makeRequests(analyzeStrategy, arg_list) 
[pool.putRequest(req) for req in requests] 
pool.wait() 


# In[18]:


i = 7
offStart = offStarts[i]
strategy_name = strategy_names[i]


# In[20]:


timeAsFloat, timeLabels, maxM, dayOff, dayTimeAsFloat = rschLib.getTimeLabels(maxD)
R = open_mtx[:, 1:]/close_mtx[:,:-1]-1  #使用收盘到开盘的回报率来修正分红和拆股
R = np.hstack((np.zeros((R.shape[0],1)), R))
tradesUsed, r_withnan = rschLib.getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat, R)
# get trade samples by good/bad trades
tradeArea=[inTime,otTime]
idxTradable = np.isfinite(r_withnan[:,tradeArea[0]])
r = r_withnan.copy()
r[np.isfinite(r)==False]=0
result = rschLib.getTradeAnalysisSampleGroups(r, idxTradable, tradeArea)

# draw price change
rschLib.drawPriceChange(r[idxTradable,:], strategy_name, timeLabels=timeLabels, tp=tradeArea)
rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
#rschLib.drawPriceChange(result['rGood20'], strategy_name, timeLabels=timeLabels, title='盈利前20%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易',  tp=tradeArea)
#rschLib.drawPriceChange(result['rBad20'], strategy_name, timeLabels=timeLabels, title='亏损前20%交易',  tp=tradeArea)
rschLib.drawPriceChange(result['rBad30'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易',  tp=tradeArea)

# analyze tags
#rschLib.analyzeTradeTags(tradesUsed, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, tkrs, offStart)
#rschLib.analyzeTradeTags(tradesUsed, result['rGood20'], result['idxGood20'], '盈利前20%交易',strategy_name, dtes, tkrs, offStart)
#rschLib.analyzeTradeTags(tradesUsed, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, tkrs, offStart)
#rschLib.analyzeTradeTags(tradesUsed, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, tkrs, offStart)
#rschLib.analyzeTradeTags(tradesUsed, result['rBad20'], result['idxBad20'], '亏损前20%交易',strategy_name, dtes, tkrs, offStart)
#rschLib.analyzeTradeTags(tradesUsed, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, tkrs, offStart)

#get tag names
tnames, tagNamesEn,t2 = rschLib.getTagNames()
idxOverLapTagList=rschLib.analyzeTradeTags(tradesUsed, r, list(range(len(tradesUsed))), '所有交易',strategy_name, dtes, tkrs, offStart, "d:\\pklWeeklyUpdate\\")

#draw pnl and tag pnl
importlib.reload(rschLib)
[dtesByTrade, pnlByTrade] = rschLib.getPnlFast(r, dtes, tkrs, name, tradesUsed, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
[dtesPnlAggr,pnlAggr, numTrades] = rschLib.aggregatePnlAndDtes(dtesByTrade, pnlByTrade)
rschLib.drawPNL(dtesPnlAggr, pnlAggr, dtes, strategy_name,showFigure='no', toDatabase='yes')
for i in range(len(tnames)):
    tagName = tnames[i]
    [dtesWithTag, pnlWithTag,n] = rschLib.aggregatePnlAndDtes(dtesByTrade[idxOverLapTagList[i]],pnlByTrade[idxOverLapTagList[i]])
    rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name, showFigure='no', toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType=tagName)
    rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name+'+'+tagNamesEn[i], showFigure='no',  toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType='pnl')

#analysis of number of trades vs performance
importlib.reload(rschLib)
rschLib.pnlVsNumtrades(pnlAggr, numTrades, strategy_name, toDatabase='yes')
rschLib.saveOffStart(strategy_name, offStart)

