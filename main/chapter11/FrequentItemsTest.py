import numpy as np
import pandas as pd

'''
使用Apriori挖掘频繁项集原理:
    对数据集中的每条交易记录transaction
    对每个候选项集can:
      检查can是否是transaction的子集：
      如果是，增加can的计数
    对每个候选项集:
      如果支持度不低于最小值，则保留该项集
      返回所有频繁项集列表
'''

# 函数1：创建一个用于测试的简单数据集
def loadDataSet():
    dataSet = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    return dataSet

# 函数2：构建第一个候选集合C1
"""
函数功能：生成第一个候选集合C1
参数说明：
dataSet：原始数据集
返回：
frozenset形式的候选集合C1
"""
def createC1(dataSet):
    C1 =[]
    for transaction in dataSet:     #数据集中每一条事务
        for item in transaction:    #事务中的每一个项集
            if not {item} in C1:    #判断每一个项集在不在候选集合中
                C1.append({item})   #添加不在候选集中的那些项集
    C1.sort() #排序                  #.sort()函数起到排序的作用，默认从小到大排序
    return list(map(frozenset, C1)) #map()是python内置的高阶函数，它接收一个函数f 和一个列表list，它的功能是把函数 f 依次作用到list 的每个元素上，得到一个新的list然后返回

#函数3：生成满足最小支持度的频繁项集L1。
"""
函数功能：
    生成满足最小支持度的频繁项集L1
参数说明:
    D:原始数据集
    Ck:候选项集
    minSupport:最小支持度
返回：
    retList：频繁项集
    supportData：所有候选项集的支持度
"""
def scanD(D,Ck,minSupport):
    ssCnt = {}              #字典的形式存放项集及其出现次数，{项集:次数}
    for tid in D:           #对于每一条事务
        for can in Ck:      #考察候选项集每一个项集
            if can.issubset(tid): #如果该项集是当前事务项的一个子集
                ssCnt[can] = ssCnt.get(can,0)+1     #项集：出现次数+1
    numItems = float(len(D))   #获取数据集的事务数
    retList=[]                 #初始化返回的频繁项集
    supportData={}             #初始化返回的所有候选项集的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        supportData[key] = support
        if support >= minSupport:
            retList.append(key)
    return retList,supportData

#函数4, 由频繁k项集生成候选k+1项集
'''
要注意的是：
    1.Ck中不会放重复的项集，这样会增加计算量。也就是说在生成候选k+1项集之后需要判断，这个候选k-1项集是
否已经存在Ck中，如果没有的话再把它放入Ck，反之则舍弃。
    2.判断两两合并之后得到的项集是不是长度为k+1,如果是则放入Ck，反之则舍弃。
'''
"""
函数功能：
    由频繁k项集生成候选k+1项集
参数说明：
    Lk：频繁k项集
返回：
    Ck：候选k+1项集
"""
def aprioriGen(Lk):
    Ck= []
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = Lk[i]
            L2 = Lk[j]
            C = L1|L2
            if not C in Ck and (len(C)==len(Lk[0])+1):  #l1|l2 == l2|l1 !!!!Set格式
                Ck.append(C)
    return Ck


#函数5 Apriori主函数
"""
Apriori主函数：
    根据输入数据集，返回所有频繁项集和所有项集及其支持度
参数说明：
    D：原始数据集
    minSupport：最小支持度
返回：
    L：所有频繁项集
    supportData：所有项集及其支持度
"""
def apriori(D,minSupport):
    C1 = createC1(D)        #初始化C1候选项集
    retList, supportData = scanD(D, C1, minSupport) #针对C1过滤，得到频繁项集和支持度列表
    L = [retList]           #定义一个容器，用于存放每一步得到的频繁项集
    k = 2
    while(len(L[-1])>0):    #以L最后一个元素是否是空为准
        Ck = aprioriGen(L[-1])  #以上一步获取的频繁项集，处理得到下一步要用的候选项集，此时每个项集的项数+1
        Lk, supK = scanD(D, Ck, minSupport) #重复处理
        supportData.update(supK)            #更新
        L.append(Lk)  # 更新频繁项集L
        k = k + 1
    return L,supportData


# dataSet = loadDataSet()
# L,supportData = apriori(dataSet,0.5)
# print(L)
# print(supportData)

