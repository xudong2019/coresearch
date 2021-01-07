#!/usr/bin/env python
# coding: utf-8

# In[8]:


strategy_name = '培宏量化1号'
inTime = 241
otTime = 600
tradeArea = [inTime, otTime]
maxD = 4
offStart = ('close_mtx',0)


# In[9]:


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
dbt = rschLib.db_tinySoftData()
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx()
# get time labels
timeAsFloat, timeLabels, maxM, dayOff, dayTimeAsFloat = rschLib.getTimeLabels(maxD)
# get trades
R = open_mtx[:, 1:]/close_mtx[:,:-1]-1  #使用收盘到开盘的回报率来修正分红和拆股
R = np.hstack((np.zeros((R.shape[0],1)), R))
tradesUsed, r_withnan = rschLib.getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat, R)

# get trade samples by good/bad trades
tradeArea=[inTime,otTime]
r = r_withnan.copy()
r[np.isfinite(r)==False]=0

# draw price change
idxTradable = np.isfinite(r_withnan[:,tradeArea[0]])
result = rschLib.getTradeAnalysisSampleGroups(r, idxTradable, tradeArea)
rschLib.drawPriceChange(r[idxTradable,:], strategy_name, timeLabels=timeLabels, tp=tradeArea)
rschLib.drawPriceChange(result['rGood10'], strategy_name, timeLabels=timeLabels, title='盈利前10%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rGood30'], strategy_name, timeLabels=timeLabels, title='盈利前30%交易', tp=tradeArea)
rschLib.drawPriceChange(result['rBad10'], strategy_name, timeLabels=timeLabels, title='亏损前10%交易',  tp=tradeArea)
rschLib.drawPriceChange(result['rBad30'], strategy_name, timeLabels=timeLabels, title='亏损前30%交易',  tp=tradeArea)
rschLib.drawPriceChange(result['rDieting'], strategy_name, timeLabels=timeLabels, title='第一天跌停', tp=tradeArea)

# analyze tags
rschLib.analyzeTradeTags(tradesUsed, result['rGood10'], result['idxGood10'], '盈利前10%交易',strategy_name, dtes, tkrs, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rGood30'], result['idxGood30'], '盈利前30%交易',strategy_name, dtes, tkrs, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rBad10'], result['idxBad10'], '亏损前10%交易',strategy_name, dtes, tkrs, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rBad30'], result['idxBad30'], '亏损前30%交易',strategy_name, dtes, tkrs, offStart)
rschLib.analyzeTradeTags(tradesUsed, result['rDieting'], result['idxDieting'], '亏损前30%交易',strategy_name, dtes, tkrs, offStart)

# get tag names
tnames, t1,t2 = rschLib.getTagNames()
idxOverLapTagList=rschLib.analyzeTradeTags(tradesUsed, r, list(range(len(tradesUsed))), '所有交易',strategy_name, dtes, tkrs, offStart)

# draw pnl and tag pnl
importlib.reload(rschLib)
[dtesByTrade, pnlByTrade] = rschLib.getPnlFast(r, idxTradable, dtes, tkrs, name, tradesUsed, inTime, otTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
[dtesPnlAggr, pnlAggr, numTrades] = rschLib.aggregatePnlAndDtes(dtesByTrade, pnlByTrade)
rschLib.drawPNL(dtesPnlAggr, pnlAggr, dtes, strategy_name, showFigure='yes', toDatabase='yes')
for i in range(len(tnames)):
    tagName = tnames[i]
    [dtesWithTag, pnlWithTag,n] = rschLib.aggregatePnlAndDtes(dtesByTrade[idxOverLapTagList[i]],pnlByTrade[idxOverLapTagList[i]])
    rschLib.drawPNL(dtesWithTag, pnlWithTag, dtes, strategy_name, showFigure='yes', toDatabase='yes', dateStart=dtesPnlAggr[0], pnlType=tagName)
    # 求非涨停，以及伴随有tagName标签的交易，绘制价格变化曲线
    idxTradableAndHasTag = list(np.intersect1d(np.nonzero(idxTradable)[0], np.array(idxOverLapTagList[i])))
    rschLib.drawPriceChange(r[idxTradableAndHasTag,:], strategy_name, timeLabels=timeLabels, title=tagName,  tp=tradeArea)

    
# control group and optimal group
[rawInTime, rawOtTime] = rschLib.getDefaultTradeTime(offStart)
idxTradableRaw = np.isfinite(r_withnan[:, rawInTime])
[dtesByTradeRaw, pnlByTradeRaw] = rschLib.getPnlFast(r, idxTradableRaw, dtes, tkrs, name, tradesUsed, rawInTime, rawOtTime, dayOff, timeAsFloat, toDatabase='yes', strategy_name=strategy_name)
[dtesPnlAggrRaw, pnlAggrRaw, numTradesRaw] = rschLib.aggregatePnlAndDtes(dtesByTradeRaw, pnlByTradeRaw)
rschLib.drawPNL(dtesPnlAggrRaw, pnlAggrRaw, dtes, strategy_name, showFigure='yes', toDatabase='yes', pnlType='rawPnl')
[bestInTime, bestOtTime] = rschLib.getOptimalTradeTime(r[idxTradable, :], rawInTime)
idxTradableBestInTime = np.isfinite(r_withnan[:, bestInTime])
[dtesByTradeBest, pnlByTradeBest] = rschLib.getPnlFast(r, idxTradableBestInTime, dtes, tkrs, name, tradesUsed, bestInTime, bestOtTime, dayOff, timeAsFloat, toDatabase='no', strategy_name=strategy_name)
[dtesPnlAggrBest, pnlAggrBest, k] = rschLib.aggregatePnlAndDtes(dtesByTradeBest, pnlByTradeBest)
rschLib.drawPNL(dtesPnlAggrBest, pnlAggrBest, dtes, strategy_name, showFigure='yes', toDatabase='yes', pnlType='optimalTradeTimePnl')
rschLib.updateStrategyBackTest(strategy_name, 'bestInTimePoint', int(bestInTime))
rschLib.updateStrategyBackTest(strategy_name, 'bestOtTimePoint', int(bestOtTime))
rschLib.updateStrategyBackTest(strategy_name, 'bestInTimeLabel', timeLabels[bestInTime])
rschLib.updateStrategyBackTest(strategy_name, 'bestOtTimeLabel', timeLabels[bestOtTime])
rschLib.updateStrategyBackTest(strategy_name, 'rawInTimePoint', int(rawInTime))
rschLib.updateStrategyBackTest(strategy_name, 'rawOtTimePoint', int(rawOtTime))
rschLib.updateStrategyBackTest(strategy_name, 'rawInTimeLabel', timeLabels[rawInTime])
rschLib.updateStrategyBackTest(strategy_name, 'rawOtTimeLabel', timeLabels[rawOtTime])

# analysis of number of trades vs performance
importlib.reload(rschLib)
rschLib.pnlVsNumtrades(pnlAggr, numTrades, strategy_name, toDatabase='yes')
# %load_ext line_profiler
# #%lprun -f getPnl getPnl()


# In[ ]:


#%lprun -f getPnl getPnl()


# In[ ]:





# In[ ]:




