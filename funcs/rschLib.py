import numpy as np
import re
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymongo
import datetime

def zero2pre(M):
    for i in range(M.shape[0]):
        for j in range(1,M.shape[1]):
            if M[i,j]==0:
                M[i,j] = M[i,j-1]
    return M

def clear_output():
    """
    clear output for both jupyter notebook and the console
    """
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    if is_in_notebook():
        from IPython.display import clear_output as clear
        clear()
        
def is_in_notebook():
    import sys
    return 'ipykernel' in sys.modules
    
def drawPriceChange(r, strategy_name, timeLabels, title='priceChange', tp=[240,604]):
    client = pymongo.MongoClient('localhost', 27017)
    db = client.quanLiang
    dbt = client.tinySoftData
    priceChange = np.mean(r,axis=0)
    priceChangeStd = np.std(r, axis=0)
    priceChangeStd[np.isfinite(priceChangeStd)==False]=0
    plt.figure()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    default_dpi = plt.rcParamsDefault['figure.dpi']
    plt.rcParams['figure.dpi'] = default_dpi*2
    plt.title('平均价格随时间变化图')
    plt.title(title)
    legends=[]
    if np.max(tp)<600:
        linewidth=1
    else:
        linewidth=0.25
    for k in tp:
        plt.plot([k, k],[-0.023,0.013], linewidth=linewidth)
        legends.append(timeLabels[k])
    plt.plot(np.cumsum(priceChange),'b-', marker="o", linewidth=0.25,markersize=0.25)
    plt.plot(priceChangeStd, 'k',  linewidth=0.25, markersize=0.25)
    legends.append('分钟价格回报率变化累积')
    legends.append('分钟价格回报率标准差')
    plt.legend(legends,bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.)
    plt.grid()
    db.strategyMinuteBar.update_one({'strategy_name':strategy_name}, {'$set':{
        'labels':timeLabels,
        title: list(np.cumsum(priceChange)),
        title+'Std': list(priceChangeStd),
        'concernPoints': tp,
        'priceType': '开盘价'
    }},upsert=True)

def getPnl(dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat,  toDatabase='no'):
    client = pymongo.MongoClient('localhost', 27017)
    db = client.quanLiang
    dbt = client.tinySoftData
    dd = list(dtes)
    nn = list(name)
    pnl = []
    dtesPnl = []
    l = [x['dateIn'] for x in trades]
    c = {}
    for q in l:
        c[q]=l.count(q)
    for (i,t) in enumerate(trades):
        inPosition = 1
        otPosition = 1
        n = t['name']
        j = dd.index(t['dateIn'])
        jin = j + dayOff[inTime]
        jot = j + dayOff[otTime]
        if (jin>=len(dtes)):
            din = '-1'
            dot = '-1'
            inPosition = 0
            otPosition = 0
            q1={
            'open':float(-1.0),
            }
            q2={
            'open':float(-1.0),
            }
        else:
            din = dtes[jin]+timeAsFloat[inTime]
            dtIn=datetime.datetime.strptime(str(din), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
            q1 = dbt.minuteBarStock.find_one({'ticker':tkrs[nn.index(n)], 'dateTime':dtIn},{'ticker':1, 'name':1, 'close':1,'open':1})
            print(q1)
            if q1==None:
                q1={'open':float(-1)}
                q2={'open':float(-1)}
            else:
                if (jot>=len(dtes)):
                    dot = '-1'
                    otPosition = 0
                    q2={
                        'open':q1['open']
                    }
                else:
                    dot = dtes[jot]+timeAsFloat[otTime]
                    dtOt=datetime.datetime.strptime(str(dot), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
                    q2 = dbt.minuteBarStock.find({'ticker':tkrs[nn.index(n)], 'dateTime':{'$gte':dtOt}},{'ticker':1, 'name':1, 'close':1,'open':1}).limit(1)[0]
                    print(q2)
                    if q2==None:
                        q2={
                            'open':q1['open']
                        }
        r = q2['open']/q1['open']-1
        print(t['dateIn'], n, din, '价',q1['open'], dot,'价',q2['open'],'利润',np.round(r*1e4)/1e2,'%', '已经开仓', inPosition, '已经平仓', otPosition)
        pnl.append(r)
        dtesPnl.append(t['dateIn'])
        if (toDatabase == 'yes'):
            db.strategyBackTestTrades.update_one({
                        '_id':t['_id']}, {'$set':{
                        '买入价':np.round(q1['open'], 2),
                        '买入时间':din,
                        '卖出价':np.round(q2['open'], 2),
                        '卖出时间':dot,
                        '利润':np.round(r,4),
                        '已经开仓':int(inPosition),
                        '已经平仓':int(otPosition)
                        }
                        })
    dq = []
    pq = []
    nq = []
    pnl = np.array(pnl)
    dtesPnl = np.array(dtesPnl)
    for x in sorted(set(dtesPnl)):
        dq.append(x)
        nq.append(np.sum(dtesPnl==x))
        pq.append(np.mean(pnl[dtesPnl==x]))
    return dq,pq,nq

def dtes2Label(dtes):
    return np.array([datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in dtes])

def drawPNL(dtesPnl,pnl,dtes, strategy_name, toDatabase='no'):
    client = pymongo.MongoClient('localhost', 27017)
    db = client.quanLiang
    dbt = client.tinySoftData
    s1 = np.nonzero(dtes>=dtesPnl[0])[0][0]
    #s2 = np.nonzero(dtes<=dtesPnl[-1])[0][-1]
    d = dtes[s1:]
    r = np.zeros(d.shape)
    l = list(d)
    for (i, x) in enumerate(dtesPnl):
        r[l.index(x)]=pnl[i]
    plt.figure()
    r = np.array(r)
    plt.plot(dtes2Label(d), np.cumsum(r))
    m = np.mean(r)
    m2 = np.mean(r[r!=0])
    v = np.std(r)
    v2 = np.std(r[r!=0])
    s = m/v*np.sqrt(250)
    plt.grid()      
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.title('策略平均日回报率:'+str(np.round(m*1e4)/1e2)+'%, 平均每笔交易回报率:'+str(np.round(m2*1e4)/1e2)\
              +'%,平均波动:'+str(np.round(v,2)*1e2)+'%, 平均每笔交易波动率:'+str(np.round(v2,2)*1e2)+'%, sharpe值:'+str(np.round(s,2)))
    if (toDatabase=='yes'):
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'labels':list([int(x) for x in d]),'pnl':list(np.cumsum(r))}},upsert=True)

def pnlVsNumtrades(pnl, numTrades, toDatabase='no'):
    plt.figure()
    q = {}
    for x in set(numTrades):
        q[x]=np.mean(np.array(pnl)[numTrades==x])
    plt.scatter(numTrades, pnl)
    plt.plot(list(q.keys()),list(q.values()),'r-')
    plt.grid()
    plt.title('成交笔数与平均回报率关系')
    if (toDatabase=='yes'):
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'numTrades':list(numTrades),'pnl':list(np.cumsum(r))}},upsert=True)
