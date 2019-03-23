import feedparser
from chapter04.bayes import *

def calcMostFreq(vocabList, fullText):
    """
    Function：   计算出现频率

    Args：       vocabList：词汇表
                fullText：全部词汇

    Returns：    sortedFreq[:30]：出现频率最高的30个词
    """
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    """
    Function：   RSS源分类器

    Args：       feed1：RSS源
                feed0：RSS源

    Returns：    vocabList：词汇表
                p0V：类别概率向量
                p1V：类别概率向量
    """
    import feedparser
    #初始化数据列表
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    #导入文本文件
    for i in range(minLen):
        #切分文本
        wordList = textParse(feed1['entries'][i]['summary'])
        #切分后的文本以原始列表形式加入文档列表
        docList.append(wordList)
        #切分后的文本直接合并到词汇列表
        fullText.extend(wordList)
        #标签列表更新
        classList.append(1)
        #切分文本
        wordList = textParse(feed0['entries'][i]['summary'])
        #切分后的文本以原始列表形式加入文档列表
        docList.append(wordList)
        #切分后的文本直接合并到词汇列表
        fullText.extend(wordList)
        #标签列表更新
        classList.append(0)
    #获得词汇表
    vocabList = createVocabList(docList)
    #获得30个频率最高的词汇
    top30Words = calcMostFreq(vocabList, fullText)
    #去掉出现次数最高的那些词
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []
    #随机构建测试集，随机选取二十个样本作为测试样本，并从训练样本中剔除
    for i in range(20):
        #随机得到Index
        randIndex = int(random.uniform(0, len(trainingSet)))
        #将该样本加入测试集中
        testSet.append(trainingSet[randIndex])
        #同时将该样本从训练集中剔除
        del(trainingSet[randIndex])
    #初始化训练集数据列表和标签列表
    trainMat = []; trainClasses = []
    #遍历训练集
    for docIndex in trainingSet:
        #词表转换到向量，并加入到训练数据列表中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        #相应的标签也加入训练标签列表中
        trainClasses.append(classList[docIndex])
    #朴素贝叶斯分类器训练函数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    #初始化错误计数
    errorCount = 0
    #遍历测试集进行测试
    for docIndex in testSet:
        #词表转换到向量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        #判断分类结果与原标签是否一致
        if classfyNb(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            #如果不一致则错误计数加1
            errorCount += 1
            #并且输出出错的文档
            print("classification error",docList[docIndex])
    #打印输出信息
    print('the erroe rate is: ', float(errorCount)/len(testSet))
    #返回词汇表和两个类别概率向量
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    """
    Function：   最具表征性的词汇显示函数

    Args：       ny：RSS源
                sf：RSS源

    Returns：    打印信息
    """
    import operator
    #RSS源分类器返回概率
    vocabList, p0V, p1V=localWords(ny, sf)
    #初始化列表
    topNY = []; topSF = []
    #设定阈值，返回大于阈值的所有词，如果输出信息很多，就提高一下阈值
    for i in range(len(p0V)):
        if p0V[i] > -4.5 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -4.5 : topNY.append((vocabList[i], p1V[i]))
    #排序
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    #打印
    for item in sortedSF:
        print(item[0])
    #排序
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    #打印
    for item in sortedNY:
        print(item[0])









# ny = feedparser.parse("https://newyork.craigslist.org/search/res?format=rss")
# print(ny)