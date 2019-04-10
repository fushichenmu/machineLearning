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


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练量
    numWords = len(trainMatrix[0])  # 每个训练样本词数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 训练类别除以训练量 = 侮辱性文档的概率P(ci)！ 这个地方trainCategory[i] =1/0
    p0Num = ones(numWords)  # 构造1*numWords向量
    p1Num = ones(numWords)  # 初始化求概率的分子变量和分母变量，
    p0Denom = 2.0
    p1Denom = 2.0  # 这里防止有一个p(xn|1)为0，则最后的乘积也为0，所有将分子初始化为1，分母初始化为2。
    for i in range(numTrainDocs):  # 遍历每一个训练文档
        if trainCategory[i] == 1:  # 如果这个文档属于侮辱性文档
            p1Num += trainMatrix[i]  # 向量相加
            p1Denom += sum(trainMatrix[i])  # 这个是分母，把trainMatrix[2]中的值先加起来为3,再把所有这个类别的向量都这样累加起来，这个是计算单词总数目
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # 对每个类别的每个单词的数目除以该类别总数目得条件概率
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classfyNb(vecClassify, p0Vec, p1Vec,
              pClass1):  # 输入是要分类的向量，使用numpy数组计算两个向量相乘的结果，对应元素相乘，然后将词汇表中所有值相加，将该值加到类别的对数概率上。比较分类向量在两个类别中哪个概率大，就属于哪一类
    p1 = sum(vecClassify * p0Vec) + log(pClass1)
    p0 = sum(vecClassify * p1Vec) + log(1 - pClass1)
    if (p1 > p0):
        return 1
    else:
        return 0


# 朴素贝叶斯词袋模型，与词集模型不同，如果一个词在文档中出现不止一次，这意味着包含该词是否出现在文档中所不能表达的某种信息。
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():  # 这是一个测试函数
    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    trainMat = []
    for item in postingList:
        trainMat.append(setOfWords2Vec(vocabList, item))
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, classVec)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(vocabList, testEntry)
    print(str(testEntry) + "  classified as:" + str(classfyNb(thisDoc, p0Vect, p1Vect, pAbusive)))
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(vocabList, testEntry)
    print(str(testEntry) + "  classified as:" + str(classfyNb(thisDoc, p0Vect, p1Vect, pAbusive)))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString) #
    return [token.lower() for token in listOfTokens if len(token) > 2] #


def spamTest():
    docList = [] # 邮件内容集，矩阵向量
    classList = [] # 标签，每个索引下的值，标识该邮件是不是垃圾邮件
    fullText = [] # 所有文件内容集合，就一个大向量
    for i in range(1, 26):
        # 获取每个ham下的文件内容
        wordList = textParse(open("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch04\\email\\spam\\%s.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch04\\email\\ham\\%s.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(wordList) #创建词汇表
    traningSet = list(range(50))  #选出50个样本
    testSet = []
    for i in range(10): # 从50个样本中挑出10个倒霉鬼做为测试样本
        randIndex =  int(random.uniform(0,len(traningSet)))
        testSet .append(traningSet[randIndex])
        del(traningSet[randIndex])
    # 那就剩下40个咯，那这40个做训练样本呗，要转换一下
    traningMat = []
    trainLabels = []
    for docIndex in traningSet:
        traningMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainLabels.append(classList[docIndex])
    # 计算条件概率
    p0V,p1V,pSpam = trainNB0(array(traningMat),array(trainLabels))
    # 校验成功率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if(classfyNb(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]):
            errorCount +=1
            print("error classify email is:%f.txt " % ((docIndex+1)/2.0))
            print (docList[docIndex])
    print("the error rate is:  "+str(errorCount/float(len(testSet))))

def spamTest_2():
    docList=[];classList = [];fullText = [];
    for i in range(1,26):#在文件夹中只有25个文件
        wordList = textParse(open("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch04\\email\\spam\\%s.txt" % i).read())
        docList.append(wordList)#把每个文本文档都加入到一个大的矩阵里面
        fullText.extend(wordList)#这个是把文本文档中所有元素都拉出来，做一个大向量
        classList.append(1)#在这个文件夹中，类别标签都是1
        wordList = textParse(open("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch04\\email\\ham\\%s.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)#统计在矩阵中出现的单词，做一个向量
    trainingSet = list(range(50));testSet=[]
    for i in range(10):#就是随机挑出10个做为测试文本，其余的作为训练样本
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses = []
    for docIndex in trainingSet:#docIndex是0-50的数，去掉了作为测试文档的10个随机数字
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))#组成训练大矩阵
        trainClasses.append(classList[docIndex])#训练大矩阵中每个向量的标签
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))#得到概率
    errorCount = 0
    for docIndex in testSet:#测试这十个测试文档的类别并于给定的类别比较以得出分类错误率
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classfyNb(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            # print "error classify email is:%f.txt " % ((docIndex+1)/2.0)
            # print docList[docIndex]
    print("the error rate is:  " + str(errorCount / float(len(testSet))))




# testingNB()
# postingList, classVec = loadDataSet()
# vocabList = createVocabList(postingList)
# print(bagOfWords2VecMN(vocabList,postingList[0]))
spamTest()