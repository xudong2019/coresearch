import numpy as np
import re
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymongo
import datetime
from pymongo import UpdateOne,InsertOne


def np_fill_zero(array: np.array) -> np.array:
    """矩阵nan和inf填0
    """
    array[np.isinf(array)] = 0
    array[np.isnan(array)] = 0
    return array

def db_quanLiang():
    db = pymongo.MongoClient('mongodb://admin:admin2020!@172.19.17.43:27018/quanLiang').quanLiang
    return db

def db_tinySoftData():
    dbt = pymongo.MongoClient('localhost').tinySoftData
    return dbt
def db_cache():
    dbc = pymongo.MongoClient('localhost').cache
    return dbc

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
    db = db_quanLiang()
    dbt = db_tinySoftData()
    r[np.isfinite(r)==False] = 0
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
        'priceType': '开盘价',
        '买入时间': timeLabels[tp[0]],
        '卖出时间': timeLabels[tp[1]],
    }},upsert=True)
    db.strategyBackTest.update_one({'strategy_name':strategy_name}, {'$set':{
        '买入时间': timeLabels[tp[0]],
        '卖出时间': timeLabels[tp[1]],
    }},upsert=True)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:55%。价格聚合分析完成。 '+str(datetime.datetime.now()),55)


#需要改为使用r来计算pnl
def getPnlFast(r, dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat, toDatabase='no', strategy_name='交易明细'):
    db = db_quanLiang()
    dbt = db_tinySoftData()
    timeAsFloatArr = np.array(timeAsFloat)
    dd = dict(zip(dtes, range(len(dtes))))
    nn = list(name)
    pnl = np.zeros(len(trades))
    dtesPnl = np.zeros(len(trades),dtype=int)
    #l = [x['dateIn'] for x in trades]
    #c = {}
    #for q in l:
    #    c[q]=l.count(q)
    bulkList = []
    for (i,t) in enumerate(trades):
        re = np.prod(1+r[i, inTime+1:otTime+1])-1
        pnl[i]=re
        if (i%10000==0):
            print(strategy_name, 'getPnl() at ', i,'/', len(trades))
        dtesPnl[i]= t['dateIn']
        if (toDatabase == 'yes')&(len(trades)-i<1e4): #交易总比数多于1万，说明筛选不够精细，前端每天显示大量的交易无意义,因此只显示1万笔以内
            inPosition = 1
            otPosition = 1
            n = t['ticker']
            j = dd[t['dateIn']]
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
                din = dtes[jin]+timeAsFloatArr[inTime]
                dtIn=datetime.datetime.strptime(str(din), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
                q1 = dbt.minuteBarStock.find_one({'ticker':n, 'dateTime':dtIn},{'open':1})
                if q1==None:
                    q1={'open':float(-1)}
                    q2={'open':float(-1)}
                    dot=-1
                else:
                    if (jot>=len(dtes)):
                        dot = '-1'
                        otPosition = 0
                        q2={
                            'open':q1['open']
                        }
                    else:
                        dot = dtes[jot]+timeAsFloatArr[otTime]
                        dtOt=datetime.datetime.strptime(str(dot), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
                        q2 = dbt.minuteBarStock.find({'ticker':n, 'dateTime':{'$gte':dtOt}},{'open':1}).limit(1)[0]
                        if q2==None:
                            q2={
                                'open':q1['open']
                            }
            #print(re, q2['open']/q1['open']-1)
            #s= str(t['dateIn'])+' '+str(n)+' '+str(din)+' '+str(q1['open'])+' '+str(dot)+' '+str(q2['open'])+' 利润 '+' '+str(np.round(re*1e4)/1e2)+'% 已经开仓'+' '+str(inPosition)+' 已经平仓'+' '+str(otPosition)+'\n'
            #f.write(s)
            #print(t['dateIn'], n, din, '价',q1['open'], dot,'价',q2['open'],'利润',np.round(re*1e4)/1e2,'%', '已经开仓', inPosition, '已经平仓', otPosition)
            bulkList.append(UpdateOne({
                        'strategy_name':strategy_name,
                        'dateIn':t['dateIn'],
                        'ticker':t['ticker'],
                        }, {'$set':{
                        '买入价':np.round(q1['open'], 2),
                        '买入时间':din,
                        '卖出价':np.round(q2['open'], 2),
                        '卖出时间':dot,
                        '利润':np.round(re,4),
                        '已经开仓':int(inPosition),
                        '已经平仓':int(otPosition)
                        }
                        }, upsert=True))
            if i%1e4==0:
                db.strategyBackTestTrades.bulk_write(bulkList)
                bulkList=[]
    if len(bulkList)>0:
        db.strategyBackTestTrades.bulk_write(bulkList)
    return dtesPnl, pnl


def getPnlWithCache(dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat,  toDatabase='no', strategy_name='交易明细'):
    timeAsFloatArr = np.array(timeAsFloat)
    #f = open(strategy_name+"交易明细.txt", "w")
    db = db_quanLiang()
    dbt = db_tinySoftData()
    dd = dict(zip(dtes, range(len(dtes))))
    nn = list(name)
    pnl = np.zeros(len(trades))
    dtesPnl = np.zeros(len(trades),dtype=int)
    #使用此函数假设是trade信息一经计算后不会改变。因此如果要改变，需要删除对应pkl文件。
    fileName = "d:\\cachePkl\\getPnl_" + strategy_name + '_' + str(inTime) + '_' +str(otTime) + ".pkl"
    if os.path.exists(fileName):
        with open(fileName, 'rb+') as f:
            pkl = pickle.load(f)
        dtesPnlInCache = pkl['dtesPnl']
        pnlInCache = pkl['pnl']
        dtesLast = dtesPnlInCache[-1]
        dtesPnl[:len(dtesPnlInCache)] = dtesPnlInCache
        pnl[:len(pnlInCache)] = pnlInCache
    else:
        dtesLast = -1
    l = [x['dateIn'] for x in trades]
    c = {}
    for q in l:
        c[q]=l.count(q)
    loc = dtesPnl
    for (i,t) in enumerate(trades):
        if t['dateIn']<=dtesLast:
            continue # 对于cache中已经计算过的pnl, 跳过
        inPosition = 1
        otPosition = 1
        n = t['ticker']
        dtesPnl[i]= t['dateIn']
        j = dd[t['dateIn']]
        jin = j + dayOff[inTime]
        jot = j + dayOff[otTime]
        if (i%10000==0):
            print('getPnl() at ', i,'/', len(trades))
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
            din = dtes[jin]+timeAsFloatArr[inTime]
            dtIn=datetime.datetime.strptime(str(din), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
            q1 = dbt.minuteBarStock.find_one({'ticker':n, 'dateTime':dtIn},{'open':1})
            if q1==None:
                q1={'open':float(-1)}
                q2={'open':float(-1)}
                dot=-1
            else:
                if (jot>=len(dtes)):
                    dot = '-1'
                    otPosition = 0
                    q2={
                        'open':q1['open']
                    }
                else:
                    dot = dtes[jot]+timeAsFloatArr[otTime]
                    dtOt=datetime.datetime.strptime(str(dot), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
                    q2 = dbt.minuteBarStock.find({'ticker':n, 'dateTime':{'$gte':dtOt}},{'open':1}).limit(1)[0]
                    if q2==None:
                        q2={
                            'open':q1['open']
                        }
        r = q2['open']/q1['open']-1
        s= str(t['dateIn'])+' '+str(n)+' '+str(din)+' '+str(q1['open'])+' '+str(dot)+' '+str(q2['open'])+' 利润 '+' '+str(np.round(r*1e4)/1e2)+'% 已经开仓'+' '+str(inPosition)+' 已经平仓'+' '+str(otPosition)+'\n'
        #f.write(s)
        #print(t['dateIn'], n, din, '价',q1['open'], dot,'价',q2['open'],'利润',np.round(r*1e4)/1e2,'%', '已经开仓', inPosition, '已经平仓', otPosition)
        pnl[i]=r
        if (toDatabase == 'yes'):
            db.strategyBackTestTrades.update_one({
                        'strategy_name':strategy_name,
                        'dateIn':t['dateIn'],
                        'ticker':t['ticker'],
                        }, {'$set':{
                        '买入价':np.round(q1['open'], 2),
                        '买入时间':din,
                        '卖出价':np.round(q2['open'], 2),
                        '卖出时间':dot,
                        '利润':np.round(r,4),
                        '已经开仓':int(inPosition),
                        '已经平仓':int(otPosition)
                        }
                        }, upsert=True)
    #f.close()
    with open(fileName, 'wb') as f:
        pickle.dump({'dtesPnl':dtesPnl, 'pnl':pnl}, f)
    return dtesPnl, pnl

def getPnl(dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat,  toDatabase='no', strategy_name='交易明细'):
    timeAsFloatArr = np.array(timeAsFloat)
    f = open(strategy_name+"交易明细.txt", "w")
    db = db_quanLiang()
    dbt = db_tinySoftData()
    dd = dict(zip(dtes, range(len(dtes))))
    nn = list(name)
    pnl = np.zeros(len(trades))
    dtesPnl = np.zeros(len(trades))
    l = [x['dateIn'] for x in trades]
    c = {}
    for q in l:
        c[q]=l.count(q)
    for (i,t) in enumerate(trades):
        inPosition = 1
        otPosition = 1
        n = t['ticker']
        dtesPnl[i]= t['dateIn']
        j = dd[t['dateIn']]
        jin = j + dayOff[inTime]
        jot = j + dayOff[otTime]
        if (i%10000==0):
            print('getPnl() at ', i,'/', len(trades))
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
            din = dtes[jin]+timeAsFloatArr[inTime]
            dtIn=datetime.datetime.strptime(str(din), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
            q1 = dbt.minuteBarStock.find_one({'ticker':n, 'dateTime':dtIn},{'open':1})
            if q1==None:
                q1={'open':float(-1)}
                q2={'open':float(-1)}
                dot=-1
            else:
                if (jot>=len(dtes)):
                    dot = '-1'
                    otPosition = 0
                    q2={
                        'open':q1['open']
                    }
                else:
                    dot = dtes[jot]+timeAsFloatArr[otTime]
                    dtOt=datetime.datetime.strptime(str(dot), '%Y%m%d.%H%M')+datetime.timedelta(hours=-8)
                    q2 = dbt.minuteBarStock.find({'ticker':n, 'dateTime':{'$gte':dtOt}},{'open':1}).limit(1)[0]
                    if q2==None:
                        q2={
                            'open':q1['open']
                        }
        r = q2['open']/q1['open']-1
        s= str(t['dateIn'])+' '+str(n)+' '+str(din)+' '+str(q1['open'])+' '+str(dot)+' '+str(q2['open'])+' 利润 '+' '+str(np.round(r*1e4)/1e2)+'% 已经开仓'+' '+str(inPosition)+' 已经平仓'+' '+str(otPosition)+'\n'
        f.write(s)
        #print(t['dateIn'], n, din, '价',q1['open'], dot,'价',q2['open'],'利润',np.round(r*1e4)/1e2,'%', '已经开仓', inPosition, '已经平仓', otPosition)
        pnl[i]=r
        if (toDatabase == 'yes'):
            db.strategyBackTestTrades.update_one({
                        'strategy_name':strategy_name,
                        'dateIn':t['dateIn'],
                        'ticker':t['ticker'],
                        }, {'$set':{
                        '买入价':np.round(q1['open'], 2),
                        '买入时间':din,
                        '卖出价':np.round(q2['open'], 2),
                        '卖出时间':dot,
                        '利润':np.round(r,4),
                        '已经开仓':int(inPosition),
                        '已经平仓':int(otPosition)
                        }
                        }, upsert=True)
    f.close()
    return dtesPnl, pnl

#dtesPnl和pnl的尺寸与trades一致，  而dtesAggr, pnlAggr, numTrades为相同日期按等权重配比计算。
def aggregatePnlAndDtes(dtesPnl, pnl):
    dtesAggr = []
    pnlAggr = []
    numTrades = []
    for x in sorted(set(dtesPnl)):
        dtesAggr.append(x)
        numTrades.append(np.sum(dtesPnl==x))
        pnlAggr.append(np.mean(pnl[dtesPnl==x]))
    return dtesAggr,pnlAggr,numTrades, 

def dtes2Label(dtes):
    return np.array([datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in dtes])

# 保存策略off_start信息。off_start信息用于对齐标签的时候避免前看效应
def saveOffStart(strategy_name, off_start):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name},{'$set':{'off_start':[off_start[0], off_start[1]]}}, upsert=True)

def updateStrategyGeneratingStatus(strategy_name, status, statusCode):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name},{'$set':{'status':status, 'statusCode':statusCode}}, upsert=True)

def drawPNL(dtesPnl,pnl,dtes, strategy_name, showFigure='no', toDatabase='no', dateStart=-1, pnlType='pnl'):
    db = db_quanLiang()
    if dateStart==-1:
        s1 = np.nonzero(dtes>=dtesPnl[0])[0][0]
    else:
        s1 = np.nonzero(dtes>=dateStart)[0][0]
    #s2 = np.nonzero(dtes<=dtesPnl[-1])[0][-1]
    d = dtes[s1:]
    r = np.zeros(d.shape)
    l = list(d)
    for (i, x) in enumerate(dtesPnl):
        r[l.index(x)]=pnl[i]
    r = np.array(r)
    m = np.mean(r)
    m2 = np.mean(r[r!=0])
    v = np.std(r)
    v2 = np.std(r[r!=0])
    s = m/v*np.sqrt(250)
    ar = np.maximum.accumulate(r) - r
    md = np.max(ar)
    dailyReturnRate = np.round(m, 5)
    tradeReturnRate = np.round(m2,5)
    dailyStd = np.round(v,5)
    tradeStd = np.round(v2,5)
    sharpe = np.round(s,2)
    mdd = np.round(md,5)
    sortino = np.round(m*250/mdd,2)
    if showFigure=='yes':
        plt.figure()
        plt.plot(dtes2Label(d), np.cumsum(r))
        plt.grid()
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        plt.title(strategy_name + ' '+pnlType+' 策略平均日回报率:'+str(dailyReturnRate*100)+'%, 平均每笔交易回报率:'+str(tradeReturnRate*100)+'%,平均波动:'+str(dailyStd*1e2)+'%, 平均每笔交易波动率:'+str(tradeStd*1e2)+'%, sharpe值:'+str(sharpe)+' 最大回撤:'+str(mdd*100))
    if (toDatabase=='yes'):
        if pnlType=='pnl':
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'labels':list([int(x) for x in d]),'pnl':list(np.cumsum(r))}},upsert=True)
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'statistics':[]}}, upsert=True)
        else:
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{pnlType:list(np.cumsum(r))}},upsert=True)
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$push':{'statistics':{
            'tag':pnlType,
            '平均日回报率':dailyReturnRate,
            '平均每笔交易回报率':tradeReturnRate,
            '平均日波动率':dailyStd,
            '平均每笔交易波动率':tradeStd,
            'Sharpe值':sharpe,
            '最大回撤':mdd,
            '索提诺比率':sortino
            }}})
            
def pnlVsNumtrades(pnl, numTrades, strategy_name, toDatabase='no'):
    db = db_quanLiang()
    dbt = db_tinySoftData()
    plt.figure()
    q = {}
    for x in set(numTrades):
        q[x]=np.mean(np.array(pnl)[numTrades==x])
    plt.scatter(numTrades, pnl)
    plt.plot(list(q.keys()),list(q.values()),'r-')
    plt.grid()
    plt.title('成交笔数与平均回报率关系')
    if (toDatabase=='yes'):
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{
        'spnlVsNumtrades':{
            'xlabel': '参与每笔交易的股票数目',
            'ylabel': '平均回报率',
            'numTrades':list([int(x) for x in numTrades]),
            'pnl':list([int(x) for x in pnl]),
            'x':list([int(x) for x in q.keys()]),
            'y':list([float(x) for x in q.values()])
            }
        }},upsert=True)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:100%。pnl。 '+str(datetime.datetime.now()),100)

#返回价格矩阵
def loadDailyBarMtx(fpath='d:\\pkl\\):
    with open(fpath+"dailyBarMtx.pkl", 'rb+') as f:
        z = pickle.load(f)
    dtes = z['dtes']
    dtes = np.array(dtes).astype(np.int)
    tkrs = z['tkrs']
    open_mtx = z['open_mtx']
    high_mtx = z['high_mtx']
    low_mtx = z['low_mtx']
    close_mtx = z['close_mtx']
    k = close_mtx==0
    close_mtx = zero2pre(close_mtx)
    open_mtx[k]=close_mtx[k]
    high_mtx[k]=close_mtx[k]
    low_mtx[k]=close_mtx[k]
    name = z['name']
    belong = z['belong']
    shenwan1 = z['shenwan1']
    shenwan2 = z['shenwan2']
    shenwan3 = z['shenwan3']
    vol_mtx  = z['vol_mtx']
    amount_mtx = z['amount_mtx']
    return dtes, tkrs, name, open_mtx, high_mtx, low_mtx, close_mtx,belong, shenwan1, shenwan2, shenwan3, vol_mtx, amount_mtx

#返回时间标签
def getTimeLabels(maxD=3):
    dbt = db_tinySoftData()
    dd = list(dbt.minuteBarStock.find({'ticker':'SH601988', 'dateAsInt':{'$gte':int(20200106), '$lte':int(20200109)}}, {'dateTime':1}))
    timeAsFloat = []
    for x in dd:
        l  = str(x['dateTime']+datetime.timedelta(hours=8))
        dl = float(l[0:4]+l[5:7]+l[8:10]+'.'+l[11:13]+l[14:16]+l[17:19])
        timeAsFloat.append(dl)
    timeAsFloat=np.round(np.array(sorted(list(set([x - int(x) for x in timeAsFloat])))), 6)
    dayTimeAsFloat=[]
    timeLabels = []
    dayOff = []
    for i in range(maxD):
        for x in timeAsFloat:
            s = str(int(x*1e4))
            if len(s)<4:
                s='0'+s
            timeLabels.append('第'+str(i+1)+'天'+s[:2]+':'+s[2:4])
            dayOff.append(i)
            dayTimeAsFloat.append(np.round(i + x, 6))
    timeAsFloat = list(timeAsFloat)*maxD
    t = np.round(np.array(sorted(list(set([x - int(x) for x in timeAsFloat])))), 6)
    maxM = t.shape[0] * maxD
    return timeAsFloat,timeLabels,maxM,dayOff,dayTimeAsFloat

#获得交易列表,如果有cache则从cache文件读取加快进度
def getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat, R):
    idxOpen = np.nonzero(np.round(dayTimeAsFloat-np.floor(dayTimeAsFloat), 4)==0.0931)[0]
    maxM = len(dayTimeAsFloat)
    dbc = db_cache()
    db = db_quanLiang()
    dbt = db_tinySoftData()
    dictDayTimeAsFloat = dict(zip(dayTimeAsFloat, range(len(dayTimeAsFloat))))
    listdtes = list(dtes)
    listtkrs = list(tkrs)
    dicttkrs = dict(zip(tkrs, range(len(tkrs))))
    dictdtes = dict(zip(dtes, range(len(dtes))))
    #使用此函数假设是一个策略的历史trade信息一经计算后不会改变。因此如果要改变，需要删除对应pkl文件。
    fileName = "d:\\cachePkl\\" + strategy_name + ".pkl"
    if os.path.exists(fileName):
        with open(fileName, 'rb+') as f:
            pkl = pickle.load(f)
        trades = pkl['trades']
        tradesUsed = pkl['tradesUsed']
        #Po = list(pkl['Po'])
        #Sale1 = list(pkl['Sale1'])
        Ret = list(pkl['Ret'])
        lastDateWithFullData = dtes[dictdtes[pkl['lastDateIn']]-maxD+1]  #比如maxD=2, 得到的trades的日期里，最后完整的数据只到maxD-1天前，所以完整数据标记只到lastDate-(maxD-1)天，之后的记录不能存入cache
        while (tradesUsed[-1]['dateIn']>lastDateWithFullData):
            print('从tradesUsed cache中踢出',tradesUsed[-1]['dateIn'])
            tradesUsed.pop()
            #Po.pop()
            Ret.pop()
            #Sale1.pop()
        while (trades[-1]['dateIn']>lastDateWithFullData):
            print('从trades cache中踢出',trades[-1]['dateIn'])
            trades.pop()
    else:
        trades = []
        tradesUsed = []
        #Po = []
        #Sale1 = []
        Ret = []
        lastDateWithFullData = 0
    loc = len(trades)
    q = list(db.strategyBackTestTrades.find({'strategy_name':strategy_name, 'dateIn':{'$gt':int(lastDateWithFullData)}},no_cursor_timeout=True).sort([('dateIn',1)]))
    trades.extend(q)
    print(strategy_name, '最后完整日期',int(lastDateWithFullData),'新增记录',len(q),'总记录',len(trades))
    lastDateIn = trades[-1]['dateIn']
    # 如果存在cache,则从cache读取记录，然后处理增量。
    bulkList = []
    for (i,x) in enumerate(trades):
        if (i<loc):
            continue
        if (x['ticker'] in listtkrs)==False:
            continue
        ticker = x['ticker']
        p = dicttkrs[x['ticker']]
        d0 = x['dateIn']
        jq = dictdtes[d0]
        j = np.min((jq+maxD-1, dtes.shape[0]-1))
        d1 = dtes[j]
        findCache = dbc.cache.find_one({'ticker':ticker, 'cacheKey':str(d0)+str(d1)})
        flag = 0 # 是否需要重新读取数据
        if findCache==None:
            flag=1
        else:
            if len(findCache['rList'])!=maxM:
                flag = 1
        if flag==1:
            dd = list(dbt.minuteBarStock.find({'ticker':ticker, 'dateAsInt':{'$gte':int(d0), '$lte':int(d1)}}, {'open':1, 'close':1, 'dateAsInt':1, 'dateTimeAsFloat':1, 'sale1':1},no_cursor_timeout=True).sort([('dateTime',1)]))
            if dd==[]:
                continue
            else:
                dq = np.unique([z['dateAsInt'] for z in dd])
                dq.sort()
                dq = dict(zip(dq, range(len(dq))))
                dt = np.array([np.round(x['dateTimeAsFloat'] - np.floor(x['dateTimeAsFloat']) + dq[x['dateAsInt']], 6) for x in dd])
                openList = np.zeros(maxM)
                closeList = np.zeros(maxM)
                sale1List = np.zeros(maxM)
                #q  = np.zeros(maxM)
                #qc = np.zeros(maxM)
                #s1 = np.zeros(maxM)
                for (i2, y) in enumerate(dd):
                    j = dictDayTimeAsFloat[dt[i2]]
                    openList[j] = y['open']
                    closeList[j] = y['close']
                    sale1List[j] = y['sale1']
                z = np.min((len(q),maxM))
                rc2o = np.zeros(maxM)
                #openList[:z] = q[:z]
                #openList[z:] = openList[z-1]
                #closeList[:z] = qc[:z]
                #closeList[z:] = closeList[z-1]
                #sale1List[:z] = s1[:z]
                #sale1List[z:] = sale1List[z-1]
                #sale1List[1:] = sale1List[:-1] #saleList第一个数据对应的时间点是9:30，但是拿的sale1是9:31的，这里以后需要改进
                # 每一分钟的开始到结束的回报率
                ro2c = closeList/openList - 1
                ro2c[ro2c==-1]=0
                # 每一分钟结束到第二分钟开始回报率
                rc2o[1:] = openList[1:]/closeList[:-1] - 1
                rc2o[rc2o==-1]=0
                k1 = dicttkrs[x['ticker']]
                k2 = dictdtes[x['dateIn']]
                maxl = np.min((k2+maxD, len(dtes)))
                rc2o[idxOpen[:maxl-k2]] = R[k1, k2:maxl] #从收盘到第二天收益率用复权数据代替
                rc2o[np.isfinite(rc2o)==False] = 0
                ro2c[np.isfinite(ro2c)==False] = 0
                rList = (1+rc2o)  # r的计算分为两部分，一个是前一分钟收盘到这一分钟开盘， 一个是前一分钟开盘到前一分钟收盘。
                rList[1:] = rList[1:] * (1+ro2c[:-1])
                rList = rList - 1
                rList[rList==-1]=0
                rList[sale1List==0] = np.nan
                if (jq+maxD-1<len(dtes)): # 从q到q+maxD-1都属于有完全数据部分，则入cache
                    bulkList.append(UpdateOne({'ticker':ticker, 'cacheKey':str(d0)+str(d1)}, {'$set':{'rList':list(rList)}}))
        else:
            #openList=findCache['openList']
            #closeList=findCache['closeList']
            #sale1List=findCache['sale1List']
            rList=findCache['rList']
        if (i%1e3==0)&(len(bulkList)>0):
            print(strategy_name, 'bulk write', ticker, d0, d1)
            dbc.cache.bulk_write(bulkList)
            bulkList = []
        if (i%1e4==0):
            print(i, '/', len(trades),d0)
        tradesUsed.append(x)
        #Po.append(openList)
        #Sale1.append(sale1List)
        Ret.append(rList)
    if len(bulkList)>0:
        dbc.cache.bulk_write(bulkList)
    #Popen = np.array(Po)
    #S = np.array(Sale1)
    r = np.array(Ret)
    lastDateIn = trades[-1]['dateIn']
    #每一分钟的开始到结束的回报率
    #ro2c = Pclos/Popen - 1
    #每一分钟结束到第二分钟开始回报率
    #rc2o = Popen[:,1:]/Pclos[:,:-1] - 1
    #rc2o = np.hstack((np.zeros((Popen.shape[0],1)),rc2o))
    #for i in range(len(tradesUsed)):
    #    k1 = dicttkrs[tradesUsed[i]['ticker']]
    #    k2 = dictdtes[tradesUsed[i]['dateIn']]
    #    maxl = np.min((k2+maxD, len(dtes)))
    #    rc2o[i, idxOpen[:maxl-k2]] = R[k1, k2:maxl]
    #rc2o[np.isfinite(rc2o)==False] = 0
    #ro2c[np.isfinite(ro2c)==False] = 0
    #r = (1+rc2o)  # r的计算分为两部分，一个是前一分钟收盘到这一分钟开盘， 一个是前一分钟开盘到前一分钟收盘。
    #r[:,1:] = r[:, 1:] * (1+ro2c[:,:-1])
    #r = r - 1
    #返回所有交易，有分钟数据交易，开盘价, 回报率
    with open(fileName, 'wb') as f:
        #pickle.dump({'trades':trades, 'tradesUsed':tradesUsed, 'Po':Po, 'lastDateIn':lastDateIn}, f)
        pickle.dump({'trades':trades, 'tradesUsed':tradesUsed, 'lastDateIn':lastDateIn, 'Ret':Ret}, f)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:35%。交易数据明细载入完成。 '+str(datetime.datetime.now()),35)
    #return trades, tradesUsed, P, r
    return tradesUsed,r

#获得交易列表,如果有cache则从cache文件读取加快进度
def getTradesWithPklCache(strategy_name, name, tkrs, dtes, maxD, maxM):
    db = db_quanLiang()
    dbt = db_tinySoftData()
    listdtes = list(dtes)
    listtkrs = list(tkrs)
    dicttkrs = dict(zip(tkrs, range(len(tkrs))))
    dictdtes = dict(zip(dtes, range(len(dtes))))
    #使用此函数假设是一个策略的历史trade信息一经计算后不会改变。因此如果要改变，需要删除对应pkl文件。
    fileName = "d:\\cachePkl\\" + strategy_name + ".pkl"
    if os.path.exists(fileName):
        with open(fileName, 'rb+') as f:
            pkl = pickle.load(f)
        trades = pkl['trades']
        tradesUsed = pkl['tradesUsed']
        Po = list(pkl['Po'])
        lastDateIn = pkl['lastDateIn']
    else:
        trades = []
        tradesUsed = []
        Po = []
        lastDateIn = 0
    loc = len(trades)
    q = list(db.strategyBackTestTrades.find({'strategy_name':strategy_name, 'dateIn':{'$gt':lastDateIn}}))
    trades.extend(q)
    lastDateIn = trades[-1]['dateIn']
    # 如果存在cache,则从cache读取记录，然后处理增量。
    for (i,x) in enumerate(trades):
        if (i<loc):
            continue
        if (x['ticker'] in listtkrs)==False:
            continue
        ticker = x['ticker']
        p = dicttkrs[x['ticker']]
        d0 = x['dateIn']
        q = dictdtes[d0]
        j = np.min((q+maxD, dtes.shape[0]-1))
        d1 = dtes[j]
        findCache = db.cache.find_one({'ticker':ticker, 'cacheKey':str(d0)+str(d1)})
        if (findCache==None):
            dd = list(dbt.minuteBarStock.find({'ticker':ticker, 'dateAsInt':{'$gte':int(d0), '$lt':int(d1)}}, {'open':1, 'dateTime':1}).sort([('dateTime',1)]))
            if dd==[]:
                continue
            else:
                db.cache.insert_one({'ticker':ticker, 'cacheKey':str(d0)+str(d1), 'value':dd})
        else:
            dd = findCache['value']
        tradesUsed.append(x)
        q = np.array([x['open'] for x in dd])
        if (i%1e4==0):
            print(i, '/', len(trades),len(q),dd[0]['dateTime'],dd[-1]['dateTime'])
        z = np.min((len(q),maxM))
        m = np.zeros(maxM)
        m[:z] = q[:z]
        m[z:] = m[z-1]
        Po.append(m)
    P = np.array(Po)
    r=P[:,1:]/P[:,:-1] - 1
    r=np.hstack((np.zeros((r.shape[0],1)),r))
    lastDateIn = trades[-1]['dateIn']
    #返回所有交易，有分钟数据交易，开盘价, 回报率
    with open(fileName, 'wb') as f:
        pickle.dump({'trades':trades, 'tradesUsed':tradesUsed, 'Po':Po, 'lastDateIn':lastDateIn}, f)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:35%。交易数据明细载入完成。 '+str(datetime.datetime.now()),35)
    return trades, tradesUsed, P, r

# 获得交易结果样本分析列表
def getTradeAnalysisSampleGroups(r, idxTradable, tradeArea):
    p = np.sum(r[:, tradeArea[0]:tradeArea[1]], axis=1)
    q = p[np.isfinite(p)]
    u = np.quantile(q,0.9)
    u2 = np.quantile(q,0.8)
    u3 = np.quantile(q,0.7)
    l = np.quantile(q,0.1)
    l2 = np.quantile(q,0.2)
    l3 = np.quantile(q,0.3)
    rGood10 = r[idxTradable&(p>=u),:]
    idxGood10 = np.nonzero(idxTradable&(p>=u))[0]
    rGood20 = r[idxTradable&(p>=u2),:]
    idxGood20 = np.nonzero(idxTradable&(p>=u2))[0]
    rGood30 = r[idxTradable&(p>=u3),:]
    idxGood30 = np.nonzero(idxTradable&(p>=u3))[0]
    rBad10 = r[idxTradable&(p<=l), :]
    idxBad10 = np.nonzero(idxTradable&(p<=l))[0]
    rBad20 = r[idxTradable&(p<=l2), :]
    idxBad20 = np.nonzero(idxTradable&(p<=l2))[0]
    rBad30 = r[idxTradable&(p<=l3), :]
    idxBad30 = np.nonzero(idxTradable&(p<=l3))[0]
    result = {}
    result['rGood10']=rGood10
    result['idxGood10']=idxGood10
    result['rGood20']=rGood20
    result['idxGood20']=idxGood20
    result['rGood30']=rGood30
    result['idxGood30']=idxGood30
    result['rBad10']=rBad10
    result['idxBad10']=idxBad10
    result['rBad20']=rBad20
    result['idxBad20']=idxBad20
    result['rBad30']=rBad30
    result['idxBad30']=idxBad30
    return result

def tagDict():
    db = db_quanLiang()
    l = list(db.tagInfo.find())
    td = {}
    for x in l:
        td[x['tagName']]={'file':x['tag'], 'off_start':(x['off_start'][0], int(x['off_start'][1]))}
    return td

def tagMapper(tagName):
    td = tagDict()
    if (tagName in td):
        return td[tagName]
    else:
        return {'file':'-1','off_start':('close_mtx', 0)}

def totInTag(checkIdx, tagMtx, dtes, tkrs, trades, offStart, tagOffStart):
    totInTag = 0
    dictd = dict(zip(dtes, range(len(dtes))))
    dictt = dict(zip(tkrs, range(len(tkrs))))
    off = 0
    idxTradesOverLapped = []
    if (offStart==('open_mtx',0)) and (tagOffStart==('close_mtx',0)):
        off = -1
    for i in checkIdx:
        b = dictd[trades[i]['dateIn']]
        a = dictt[trades[i]['ticker']]
        if tagMtx[a,b+off]==1:
            totInTag = totInTag +1
            idxTradesOverLapped.append(i)
    return totInTag/len(checkIdx), idxTradesOverLapped

#获得tagNames
def getTagNames():
    tagNames = list(tagDict().keys())
    tagNamesEn = [x['file'] for x in list(tagDict().values())]
    tagStartOffs = [(x['off_start'][0], x['off_start'][1]) for x in list(tagDict().values())]
    return tagNames, tagNamesEn, tagStartOffs

#分析trades中的checkIdx部分和所有标签的重叠交易
def analyzeTradeTags(trades, r, checkIdx, title, strategy_name, dtes, tkrs, offStart=('close_mtx',0), fpath="d:\\pkl\\"):
    db = db_quanLiang()
    dbt = db_tinySoftData()
    tagNames, tagNamesEn, tagOffStarts = getTagNames()
    tagNamesEn = [x['file'] for x in list(tagDict().values())]
    idxTradesOverLappedList = []
    s = []
    print(tagNames)
    for i in range(len(tagNames)):
        t = tagMapper(tagNames[i])
        with open(fpath + t['file'] + ".pkl", 'rb+') as f:
            tagFile = pickle.load(f)
        tagOffStart = t['off_start']
        tagMtx = tagFile['tag_mtx']
        p1, idxOverLappedTrades = totInTag(checkIdx, tagMtx, dtes, tkrs, trades, offStart, tagOffStart)
        idxTradesOverLappedList.append(idxOverLappedTrades)
        p = np.round(p1*1e4)/1e2
        print(title, tagNames[i], p)
        s.append({'标签':tagNames[i], '符合度':p})
    s.sort(key=lambda t:t['符合度'], reverse=True)
    db.strategyMinuteBar.update_one({'strategy_name':strategy_name}, {'$set':{
        title+'标签': s
    }},upsert=True)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:75%。归因分析完成。 '+str(datetime.datetime.now()),75)
    return idxTradesOverLappedList
    