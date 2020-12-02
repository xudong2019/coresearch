#!/usr/bin/env python
# coding: utf-8

# In[1]:


strategy_name ='鹏晖量化1号'


# In[2]:


import sys
sys.path.append("C:\Program Files\Tinysoft\Analyse.NET")
sys.path.append(r"C:\Users\xudong\Documents\github\web\dataServer")
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
client = pymongo.MongoClient('mongodb://admin:admin2020!@172.19.17.43:27018/quanLiang', 27017)
db = client.quanLiang


# In[3]:


def nowTime():
    return int(time.strftime("%H%M%S",time.localtime(time.time())))
def nowString():
    return time.strftime("%Y%m%d %H:%M:%S",time.localtime(time.time()))
nowTime()
nowString()


# In[4]:


def runStrategy():
    while (nowTime()>150000)|(nowTime()<92700):
        print('waiting for market open...')
        time.sleep(60)
    n = list(db.tkrsInfo.find({'TagBreak1YHighOpen':1, 'tagCirculateMarketValueBiggerThan100Y':1, 'TagPreDayBreak1YHigh':0, 'TagIsZhangting':0},{'name':1}))
    #n = list(db.tkrsInfo.find({'TagBreak1YHighOpen':1},{'name':1}))
    names = [x['name'] for x in n]
    if (len(names)>0):
        s='开盘初选：发现符合条件标的:'
        for x in names:
            s = s + x + ' '
    else:
        s='开盘初选：没有符合条件标的'
    db.strategyEventRecords.insert_one({'strategy_name':strategy_name, 'updateTime':nowString(), 'content':s})
    print('初选:', names)
    if len(names)==0:
        return
    while (nowTime()<150000):
        print('waiting for market close...')
        time.sleep(60)
    n = list(db.tkrsInfo.find({'TagIsZhangting':0, 'name':{'$in':names}},{'name':1}))
    names = [x['name'] for x in n]
    if (len(names)>0):
        s='收盘复选：可交易标的:'
        for x in names:
            s = s + x + ' '
    else:
        s='收盘复选：没有符合条件标的'
    db.strategyEventRecords.insert_one({'strategy_name':strategy_name, 'updateTime':nowString(), 'content':s})
    print(names)


# In[ ]:


runStrategy()


# In[11]:


1+1

