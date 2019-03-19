from numpy import *

# 词表到向量的转换函数：
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not 这些标注信息用于训练程序以便自动检测侮辱性留言
    return postingList, classVec


# 创建字典集，包含所有文档中出现的不重复列表
def createVocabList(dataSet):
    vocabSet = set([])  # create an empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集，详参附录C学习其他符号使用
    return list(vocabSet)


# 把每条语句，转化为一个标准向量供给分析
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #训练量
    numWords = len(trainMatrix[0]) #每个训练样本词数
    pAbusive =sum(trainCategory) / float(numTrainDocs) #训练类别除以训练量 = 侮辱性文档的概率P(ci)！ 这个地方trainCategory[i] =1/0
    p0Num = zeros(numWords) #
    p1Num = zeros(numWords) # 构造 n*n 0型矩阵
    p0Denom = 0.0;p1Denom= 0.0 #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive


# postingList, classVec = loadDataSet()
# vocabList = createVocabList(postingList)
# print(setOfWords2Vec(vocabList,postingList[0]))
