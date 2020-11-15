#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import importlib
import pickle
from pylab import mpl
import pymongo
import sys
sys.path.append(r"C:\\Users\\xudong\\Documents\\github\\coresearch\\funcs\\")
sys.path.append(r"C:\\Users\\xudong\\Documents\\guanzhao\\dataserver\\")
from library.tags import tags
import rschLib
import pickle

client = pymongo.MongoClient('localhost', 27017)
db = client.quanLiang
dbt = client.tinySoftData
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx()


# In[8]:


def generateTagAnalysis(tagName):
    tname = rschLib.tagMapper(tagName)['file']
    strategy_name = tname
    back_test_days = 500
    with open("d:\\pkl\\" + tname + ".pkl", 'rb+') as f:
        tagInfo = pickle.load(f)
    tag_mtx = tagInfo['tag_mtx']
    off_start = tagInfo['off_start']
    tagNotNew = np.zeros(close_mtx.shape)
    for i in range(close_mtx.shape[0]):
        j = np.nonzero(close_mtx[i,:]>0)[0][0]
        v = np.min((close_mtx.shape[1], j+60))
        tagNotNew[i, v:]=1
    tag_mtx = (tag_mtx==1) & (tagNotNew==1) # 去掉新股票
    max_holding_days = 10
    daily_stage = []
    matrix_types = ['open_mtx', 'close_mtx']
    flag = 0
    for i in range(max_holding_days*2):
        t = (matrix_types[i%2], int(i/2))
        if (t==off_start):
            flag = 1
        if flag==1:
            daily_stage.append((matrix_types[i%2], int(i/2)))
    matrixName = {
        'open_mtx': '开盘价',
        'close_mtx': '收盘价'
    }
    # %% 画出
    plt.figure()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    default_dpi = plt.rcParamsDefault['figure.dpi']
    plt.rcParams['figure.dpi'] = default_dpi*2
    pnlByPeriod  = [] # 在特定时间区间的持股标签平均回报率，如标签产生后的当天开盘到当天收盘，
    dtesUsedByPeriod = []
    tagsUsedByPeriod = []
    legends = []
    for i in range(len(daily_stage)-1):
        priceIn = tags.TagBase.get_zero2pre(daily_stage[i][0])[:, daily_stage[i][1]:]
        priceOt = tags.TagBase.get_zero2pre(daily_stage[i+1][0])[:, daily_stage[i+1][1]:]
        priceIn = priceIn[:, :priceOt.shape[1]]  # priceIn 和 priceOt头对齐， 如果priceOt长度小于priceIn, 则priceIn取部分
        pct_change = rschLib.np_fill_zero(priceOt / priceIn - 1) # 百分比变化
        dtesUsed = dtes[daily_stage[i][1]:]
        dtesUsed = dtesUsed[:priceOt.shape[1]]  # 对齐后的日期
        tag_mtxUsed = tag_mtx[:, :priceOt.shape[1]] # 对齐标签矩阵
        ret = tag_mtxUsed * pct_change # 回报率矩阵
        ret_avg = rschLib.np_fill_zero(ret.sum(axis=0)/tag_mtxUsed.sum(axis=0)) # 计算按tag持股平均回报率
        pnlByPeriod.append(ret_avg)
        dtesUsedByPeriod.append(dtesUsed)
        tagsUsedByPeriod.append(tag_mtxUsed)
        plt.plot(np.cumsum(ret_avg))
        legends.append('第'+str(daily_stage[i][1])+'天的'+matrixName[daily_stage[i][0]])
    pnlByPeriod_mtx = np.zeros((len(pnlByPeriod), len(pnlByPeriod[-1])))
    plt.title(tagName+'标签持股区间平均回报率')
    plt.legend(legends)
    plt.grid()

    for i in range(len(pnlByPeriod)):
        pnlByPeriod_mtx[i,:] = pnlByPeriod[i][:len(pnlByPeriod[-1])]

    # %% calculate best investment records
    best = 0
    dtesUsed = dtesUsedByPeriod[best][-back_test_days:]
    tag_mtxUsed = tagsUsedByPeriod[best][:, -back_test_days:]
    pnl = [float(x) for x in np.cumsum(pnlByPeriod[best][-back_test_days:])]
    plt.title('pnl')
    plt.plot(pnl)
    plt.grid()
    labels = [int(x) for x in dtesUsed]
    db.strategyBackTest.update_one({'strategy_name': strategy_name}, {'$set':{
        'strategy_name':strategy_name,
        'rawLabels':labels,
        'rawPnl':pnl
        }},upsert=True)
    # %% priceChange
    priceChange = np.mean(pnlByPeriod_mtx[:, -back_test_days:], axis=1)
    labels = pnlByPeriod_mtx[best][-back_test_days:]
    plt.figure()
    plt.plot(np.cumsum(priceChange))

    # %% 每笔交易上传至数据库
    db.strategyBackTestTrades.remove({'strategy_name':strategy_name})
    for (i, x) in enumerate(dtesUsed):
        q = name[tag_mtxUsed[:,i]==1]
        for y in q:
            db.strategyBackTestTrades.update({
                'strategy_name':strategy_name,
                'name':y,
                'dateIn':int(x)
            })


# In[9]:


importlib.reload(rschLib)
tagNames = list(rschLib.tagDict().keys())
for q in range(len(tagNames)):
    generateTagAnalysis(tagNames[q])


# In[ ]:




