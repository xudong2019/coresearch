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
import datetime

db = rschLib.db_quanLiang()
dbt = rschLib.db_tinySoftData()
dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx = rschLib.loadDailyBarMtx()


# In[2]:


def generateTagAnalysis(tagName):
    print(tagName)
    tname = rschLib.tagMapper(tagName)['file']
    strategy_name = tname
    startDate = 20180101
    back_test_days = 500
    with open("d:\\pkl\\" + tname + ".pkl", 'rb+') as f:
        tagInfo = pickle.load(f)
    tag_mtx = tagInfo['tag_mtx']
    off_start = rschLib.tagMapper(tagName)['off_start']
    rschLib.saveOffStart(strategy_name, off_start)
    tagNotNew = np.zeros(close_mtx.shape)
    for i in range(close_mtx.shape[0]):
        j = np.nonzero(close_mtx[i,:]>0)[0][0]
        v = np.min((close_mtx.shape[1], j+60))
        tagNotNew[i, v:]=1
    tag_mtx = (tag_mtx==1) & (tagNotNew==1) # 去掉新股票
    max_holding_days = 4
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

    # %% calculate best investment records, 如果是开盘的，假设开盘可以买入， 如果是收盘的，假设第二天开盘可以买入
    if off_start==('open_mtx', 0):
        best = 0
    else:
        best = 1
    idxStart = dtesUsedByPeriod[best]>startDate
    idxStart2 = dtesUsedByPeriod[best+1]>startDate
    dtesUsed = dtesUsedByPeriod[best][idxStart]
    tag_mtxUsed = tagsUsedByPeriod[best][:, idxStart]
    n1 = np.cumsum(pnlByPeriod[best][idxStart])
    n2 = np.cumsum(pnlByPeriod[best+1][idxStart2])
    n1[len(n1)-len(n2):] = n1[len(n1)-len(n2):]+n2
    pnl = [float(x) for x in n1]
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
    plt.figure()
    plt.plot(np.cumsum(priceChange))

    # %% 每笔交易上传至数据库
    db.strategyBackTest.remove({'strategy_name':strategy_name})
    db.strategyBackTestTrades.remove({'strategy_name':strategy_name})
    dtesUsed = [int(x) for x in dtesUsed]
    for (i, x) in enumerate(dtesUsed):
        q = name[tag_mtxUsed[:,i]==1]
        tk = tkrs[tag_mtxUsed[:,i]==1]
        query = []
        for i, y in enumerate(q):
            query.append({
                'strategy_name':strategy_name,
                'name':y,
                'ticker':tk[i],
                'dateIn':int(x)
            })
        if len(query)>0:
            db.strategyBackTestTrades.insert_many(query)
    rschLib.updateStrategyGeneratingStatus(strategy_name, '生成进度:10%。初始化标签。'+str(datetime.datetime.now()),10)


# In[3]:


importlib.reload(rschLib)
tagNames = list(rschLib.tagDict().keys())
for q in range(len(tagNames)):
    generateTagAnalysis(tagNames[q])


# In[4]:


# importlib.reload(rschLib)
# %load_ext line_profiler
# %lprun -f generateTagAnalysis generateTagAnalysis(tagName)


# In[ ]:




