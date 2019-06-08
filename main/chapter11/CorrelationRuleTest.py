from chapter11.FrequentItemsTest import *

"""
函数功能：
    根据 单个频繁项集 生成满足最小置信度的关联规则
参数说明：
    freqSet：单个频繁项集
    H：可以出现在关联规则右部的元素列表
    supportData：所有项集及其支持度
    brl：强关联规则列表
    minConf：最小置信度
返回：
    prunedH：关联规则右部元素列表
"""
def calconf(freqSet,H,supportData,brl,minconf=0.5):
    prunedH = []
    for conseq in H:           #对于每一个后件
        conf =supportData[freqSet]/supportData[freqSet-conseq]   #计算其可信度
        if conf >= minconf:                                        #如果大于要求的minconf
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))           #强关联列表.append(前件，后件，置信度）
            prunedH.append(conseq)                                 #prunedH，返回后件子集，子集有可能有满足minconf的关联规则
    return prunedH

"""
频繁多项集生成函数,对于多项集，生成置信度规则的过程会更加复杂。


函数功能：
    根据单个频繁项集生成满足最小置信度的关联规则
参数说明：
    freqSet：单个频繁项集
    H：频繁项集中的所有子项，可以放在规则右部的元素列表
    brl：存放关联规则的容器
"""
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    Hmp =True
    while Hmp:
        Hmp =False
        H =  calconf(freqSet, H, supportData, brl, minConf)
        H =aprioriGen(H)
        Hmp = not (H==[] or len(H[0]) ==len(freqSet))


"""
函数功能：
    生成所有满足最小置信度的关联规则
参数说明：
    L：频繁项集
    supportData：所有项集及其支持度
    minConf：最小置信度
返回：
    bigRuleList：满足最小置信度的关联规则（即强关联规则）
"""
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []  # 强关联规则容器
    for i in range(1,len(L)):  #单个频繁项集不考虑 对于每个k-频繁项集
        for freqSet in L[i]:   #每个频繁项集
            H1 = [frozenset([item]) for item in freqSet ]   #获取单项集合
            if(i>1):    #如果不是2项集
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)  # 频繁多项集 
            else:
                calconf(freqSet, H1, supportData, bigRuleList, minConf) #频繁二项集
    return bigRuleList


# data = open(r'C:\Users\Mypc\Desktop\第10期 Apriori算法\groceries.txt')
# gro = data.readlines()
# tran = [gro[i].strip('\n').split(" ") for i in range(len(gro))]
# L, supportData = apriori(tran, minSupport = 0.5)
# brl = generateRules(L, supportData, minConf=0.5)
# print(brl)

# dataSet = loadDataSet()
# L,supportData = apriori(dataSet,0.5)
# bigRuleList = generateRules(L,supportData,0.7)
# print(bigRuleList)
