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
    
def drawPriceChange(r, strategy_name, timeLabels, title='priceChange', tp=[240,604], showGraph='yes'):
    if len(r)==0:
        return
    db = db_quanLiang()
    dbt = db_tinySoftData()
    r[np.isfinite(r)==False] = 0
    priceChange = np.mean(r,axis=0)
    priceChangeStd = np.std(r, axis=0)
    priceChangeStd[np.isfinite(priceChangeStd)==False]=0
    if (showGraph=='yes'):
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
    if (title=='priceChange'):
        db.strategyBackTest.update_one({'strategy_name':strategy_name}, {'$set':{
            '买入时间': timeLabels[tp[0]],
            '卖出时间': timeLabels[tp[1]],
        }},upsert=True)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:55%。价格聚合分析完成。 '+str(datetime.datetime.now()),55)


#需要改为使用r来计算pnl
def getPnlFast(r, idxTradable, dtes,tkrs, name, trades, inTime, otTime, dayOff, timeAsFloat, toDatabase='no', strategy_name='交易明细'):
    db = db_quanLiang()
    dbt = db_tinySoftData()
    timeAsFloatArr = np.array(timeAsFloat)
    dd = dict(zip(dtes, range(len(dtes))))
    nn = list(name)
    pnl = np.zeros(len(trades))
    dtesPnl = np.zeros(len(trades),dtype=int)
    bulkList = []
    for (i,t) in enumerate(trades):
        if idxTradable[i]==True:
            re = np.prod(1+r[i, inTime+1:otTime+1])-1
            pnl[i]=re
        else: 
            pnl[i]=0
        if (i%100000==0):
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
                        tt1 = list(dbt.minuteBarStock.find({'ticker':n, 'dateTime':{'$gte':dtOt}},{'open':1}).limit(1))
                        if len(tt1)==0:
                            q2={
                                'open':q1['open']
                            }
                        else:
                            q2 = tt1[0]
            bulkList.append(UpdateOne({
                        'strategy_name':strategy_name,
                        'dateIn':t['dateIn'],
                        'ticker':t['ticker'],
                        }, {'$set':{
                        'name':t['name'],
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


#dtesPnl和pnl的尺寸与trades一致，  而dtesAggr, pnlAggr, numTrades为相同日期按等权重配比计算。
def aggregatePnlAndDtes(dtesPnl, pnl):
    dtesAggr = []
    pnlAggr = []
    numTrades = []
    for x in sorted(set(dtesPnl)):
        dtesAggr.append(x)
        numTrades.append(np.sum(dtesPnl==x))
        pnlAggr.append(np.mean(pnl[dtesPnl==x]))
    return dtesAggr,pnlAggr,numTrades

def dtes2Label(dtes):
    return np.array([datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in dtes])

# 保存策略off_start信息。off_start信息用于对齐标签的时候避免前看效应
def saveOffStart(strategy_name, off_start):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name},{'$set':{'off_start':[off_start[0], off_start[1]]}}, upsert=True)

def updateStrategyGeneratingStatus(strategy_name, status, statusCode):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name},{'$set':{'status':status, 'statusCode':statusCode, 'lastUpdate':str(datetime.datetime.now())}}, upsert=True)

def updateStrategyPrivacy(strategy_name, privacyStatus):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name},{'$set':{'privacy':privacyStatus}}, upsert=True)

def updateStrategyBackTest(strategy_name, field, value):
    db = db_quanLiang()
    db.strategyBackTest.update({'strategy_name':strategy_name}, {'$set':{field:value}}, upsert=True)

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
    ar = np.maximum.accumulate(np.cumsum(r)) - np.cumsum(r)
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
        statisticsInfo = {
            'tag':pnlType,
            '平均日回报率':dailyReturnRate,
            '平均每笔交易回报率':tradeReturnRate,
            '平均日波动率':dailyStd,
            '平均每笔交易波动率':tradeStd,
            'Sharpe值':sharpe,
            '最大回撤':mdd,
            '索提诺比率':sortino
            }
        if pnlType=='pnl':
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'labels':list([int(x) for x in d]),'pnl':list(np.cumsum(r))}},upsert=True)
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'statistics':[]}}, upsert=True)
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'performance':statisticsInfo}}, upsert=True)
        else:
            db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{pnlType:list(np.cumsum(r))}},upsert=True)
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$push':{'statistics': statisticsInfo}})
            
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
        db.strategyBackTest.update_one({'strategy_name':strategy_name},{'$set':{'averageTrades':np.round(np.mean(numTrades), 1)}},upsert=True)
    updateStrategyGeneratingStatus(strategy_name, '生成进度:100%。pnl。 '+str(datetime.datetime.now()),100)

#返回价格矩阵
def loadDailyBarMtx(fpath='d:\\pkl\\'):
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
    timeAsFloat2 = []
    for x in dd:
        l  = str(x['dateTime']+datetime.timedelta(hours=8))
        l2 = str(x['dateTime']+datetime.timedelta(hours=8)-datetime.timedelta(minutes=1))
        dl = float(l[0:4]+l[5:7]+l[8:10]+'.'+l[11:13]+l[14:16]+l[17:19])
        dl2 = float(l2[0:4]+l2[5:7]+l2[8:10]+'.'+l2[11:13]+l2[14:16]+l2[17:19])
        timeAsFloat.append(dl)
        timeAsFloat2.append(dl2)
    timeAsFloat=np.round(np.array(sorted(list(set([x - int(x) for x in timeAsFloat])))), 6)
    timeAsFloat2=np.round(np.array(sorted(list(set([x - int(x) for x in timeAsFloat2])))), 6) #对于采用分钟数据的open价格的，使用timeLabel2作为交易时间更准确
    dayTimeAsFloat=[]
    timeLabels = []
    timeLabels2 = []
    dayOff = []
    for i in range(maxD):
        for (j,x) in enumerate(timeAsFloat):
            s = str(int(x*1e4))
            s2 = str(int(timeAsFloat2[j]*1e4))
            if len(s)<4:
                s='0'+s
            if len(s2)<4:
                s2='0'+s2
            timeLabels.append('第'+str(i+1)+'天'+s[:2]+':'+s[2:4])
            timeLabels2.append('第'+str(i+1)+'天'+s2[:2]+':'+s2[2:4])
            dayOff.append(i)
            dayTimeAsFloat.append(np.round(i + x, 6))
    timeAsFloat = list(timeAsFloat)*maxD
    t = np.round(np.array(sorted(list(set([x - int(x) for x in timeAsFloat])))), 6)
    maxM = t.shape[0] * maxD
    return timeAsFloat,timeLabels2,maxM,dayOff,dayTimeAsFloat

#获得交易列表,如果有cache则从cache文件读取加快进度
def getTradesFast(strategy_name, name, tkrs, dtes, maxD, dayTimeAsFloat, R, skipDump=False):
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
        readFile = True
        try:
            with open(fileName, 'rb+') as f:
                pkl = pickle.load(f)
            tradesUsed = pkl['tradesUsed']
            Ret = np.array(pkl['Ret'])
            lastDateWithFullData = dtes[dictdtes[pkl['lastDateIn']]-maxD+1]  #比如maxD=2, 得到的trades的日期里，最后完整的数据只到maxD-1天前，所以完整数据标记只到lastDate-(maxD-1)天，之后的记录不能存入cache
            flagKicked=-1
            totk=0
            while (tradesUsed[-1]['dateIn']>lastDateWithFullData):
                if (flagKicked!=tradesUsed[-1]['dateIn']):
                    print('从tradesUsed cache中踢出',tradesUsed[-1]['dateIn'])
                    flagKicked = tradesUsed[-1]['dateIn']
                tradesUsed.pop()
                totk=totk+1
            if totk!=0:
                print(totk)
                Ret =Ret[:-totk,:]
        except Exception as e:
            readFile = False
            print(str(e))
    else:
        readFile = False
    if readFile == False:
        Ret = np.array([]).reshape((0, len(dayTimeAsFloat)))
        tradesUsed = []
        lastDateWithFullData = 0
    RetTemp = []
    trades = list(db.strategyBackTestTrades.find({'strategy_name':strategy_name, 'dateIn':{'$gt':int(lastDateWithFullData)}},no_cursor_timeout=True).sort([('dateIn',1)]))
    if len(trades)>2e5:
        return [],[] # 如果记录超过1百万条，暂时不处理标签
    print(strategy_name, '最后完整日期',int(lastDateWithFullData),'新增记录',len(trades),'总记录',len(tradesUsed))
    lastDateIn = trades[-1]['dateIn']
    # 如果存在cache,则从cache读取记录，然后处理增量。
    bulkList = []
    for (i,x) in enumerate(trades):
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
                print('has cache but len not right', len(findCache['rList']), maxM)
                flag = 1
            if np.nanmin(findCache['rList'])<-0.4:  # 不可能一分钟跌超过40%
                print('has cache but minimum not right', np.nanmin(findCache['rList']))
                print(x)
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
                rc2o = np.zeros(maxM)
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
                    bulkList.append(UpdateOne({'ticker':ticker, 'cacheKey':str(d0)+str(d1)}, {'$set':{'rList':list(rList)}}, upsert=True))
                    if len(bulkList)>1e3:
                        print(strategy_name, 'bulk write', ticker, d0, d1)
                        dbc.cache.bulk_write(bulkList)
                        bulkList = []
        else:
            rList=findCache['rList']
        if (i%1e4==0):
            print(i, '/', len(trades),d0)
        tradesUsed.append(x)
        RetTemp.append(rList)
    if len(bulkList)>0:
        dbc.cache.bulk_write(bulkList)
    rTemp = np.array(RetTemp)
    Ret = np.vstack((Ret, rTemp))
    lastDateIn = tradesUsed[-1]['dateIn']
    if (skipDump==False):
        try:
            with open(fileName, 'wb') as f:
                pickle.dump({'tradesUsed':tradesUsed, 'lastDateIn':lastDateIn, 'Ret':Ret}, f)
        except Exception as e:
            print(str(e))
    updateStrategyGeneratingStatus(strategy_name, '生成进度:35%。交易数据明细载入完成。 '+str(datetime.datetime.now()),35)
    return tradesUsed,Ret

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
    t2 = 240*(1+int(tradeArea[0]/240))
    rc = np.sum(r[:, tradeArea[0]:t2], axis=1)
    uDieting = rc<=-0.095
    idxDieting = np.nonzero(idxTradable&uDieting)[0]
    rDieting = r[idxTradable&uDieting,:]
    result = {}
    result['rDieting']=rDieting
    result['idxDieting']=idxDieting
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
        if (trades[i]['ticker'] in dictt)==False:
            continue
        if (trades[i]['dateIn'] in dictd)==False:
            continue
        a = dictt[trades[i]['ticker']]
        b = dictd[trades[i]['dateIn']]
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
    if len(r)==0:
        return
    db = db_quanLiang()
    dbt = db_tinySoftData()
    tagNames, tagNamesEn, tagOffStarts = getTagNames()
    tagNamesEn = [x['file'] for x in list(tagDict().values())]
    idxTradesOverLappedList = []
    s = []
    print(tagNames)
    for i in range(len(tagNames)):
        t = tagMapper(tagNames[i])
        if os.path.exists(fpath + t['file'] + ".pkl")==False:
            idxTradesOverLappedList.append([])
            continue
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

def getDefaultTradeTime(offStart):
    if offStart[0] == 'open_mtx':
        it = 1
    if offStart[0] == 'close_mtx':
        it = 240
    it = it + 240*offStart[1]
    rawInTime = it # 默认入场时间
    rawOtTime = it+240 # 默认出场时间
    return rawInTime, rawOtTime

def getOptimalTradeTime(r, firstAvailableTime):
    q = np.cumsum(np.mean(r, axis=0))
    k1 = firstAvailableTime + np.nonzero(q[firstAvailableTime:-240]==np.min(q[firstAvailableTime:-240]))[0][0]
    print(k1)
    if (k1==len(q)-1):
        k1 = fistAvailableTime
        k2 = k1+240 # 价格一路下降，最低点为最后点情况，说明没有最优价格
    else:
        km = 240*(int(k1/240)+1)
        k2 = km + np.nonzero(q[km:]==np.max(q[km:]))[0][0]
    return k1, k2 # 最好的k1和k2