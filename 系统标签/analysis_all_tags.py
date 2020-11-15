#!/usr/bin/env python
# coding: utf-8

# In[1]:


maxD = 5
inTime = 234
otTime = 474


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
db = client.quanLiang
dbt = client.tinySoftData
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx()
# get time labels
# get trades


# In[3]:


strategy_names = [x['file'] for x in rschLib.tagDict().values()]
offStarts = [x['off_start'] for x in rschLib.tagDict().values()]
importlib.reload(rschLib)


# In[4]:


def analyzeStrategy(strategy_name, offStart, dtes, name, tkrs):
    timeAsFloat, timeLabels, maxM, dayOff = rschLib.getTimeLabels(maxD)
    trades, tradesUsed, Po, r = rschLib.getTrades(strategy_name, name, tkrs, dtes, maxD, maxM)
    # get trade samples by good/bad trades
    tradeArea=[inTime,otTime]
    result = rschLib.getTradeAnalysisSampleGroups(r, tradeArea)

    # draw price change
    rschLib.drawPriceChange(r, strategy_name, timeLabels=timeLabels, tp=tradeArea)
    rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rGood20'], strategy_name, timeLabels=timeLabels, title='盈利前20%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rBad20'], strategy_name, timeLabels=timeLabels, title='亏损前20%交易', tp=tradeArea)
    rschLib.drawPriceChange(result['rBad30'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易', tp=tradeArea)
    # analyze tags
    rschLib.analyzeTradeTags(trades, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, name, offStart)
    rschLib.analyzeTradeTags(trades, result['rGood20'], result['idxGood20'], '盈利前20%交易',strategy_name, dtes, name, offStart)
    rschLib.analyzeTradeTags(trades, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, name, offStart)
    rschLib.analyzeTradeTags(trades, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, name, offStart)
    rschLib.analyzeTradeTags(trades, result['rBad20'], result['idxBad20'], '亏损前20%交易',strategy_name, dtes, name, offStart)
    rschLib.analyzeTradeTags(trades, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, name, offStart)

    #get tag names
    tnames, idxOverLapTagList=rschLib.analyzeTradeTags(trades, r, list(range(len(trades))), '所有交易',strategy_name, dtes, name, offStart)

    #draw pnl and tag pnl
    importlib.reload(rschLib)
    [dtesByTrade, pnlByTrade]=rschLib.getPnl(dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
    [dtesPnlAggr,pnlAggr, numTrades] = rschLib.aggregatePnlAndDtes(dtesByTrade, pnlByTrade)
    rschLib.drawPNL(dtesPnlAggr, pnlAggr, dtes, strategy_name, toDatabase='yes')
    for i in range(len(tnames)):
        tagName = tnames[i]
        [dtesWithTag, pnlWithTag,n] = rschLib.aggregatePnlAndDtes(dtesByTrade[idxOverLapTagList[i]],pnlByTrade[idxOverLapTagList[i]])
        rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name, toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType=tagName)
        rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name+'+'+tagName, toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType='pnl')

    #analysis of number of trades vs performance
    importlib.reload(rschLib)
    rschLib.pnlVsNumtrades(pnlAggr, numTrades, strategy_name, toDatabase='yes')


# In[ ]:


importlib.reload(rschLib)
for (i,strategy_name) in enumerate(strategy_names):
    offStart = offStarts[i]
    analyzeStrategy(strategy_name, offStart, dtes, name, tkrs)


# In[ ]:


#%load_ext line_profiler
#%lprun -f analyzeStrategy analyzeStrategy(strategy_name, offStart, dtes, name, tkrs)

