from chapter05.logsticRegression import *


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch05\\horseColicTraining.txt')
    frTest = open('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch05\\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #一条数据有22列，最后一列为类别标签
            lineArr.append(float(currLine[i]))#注意文本要转float
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscentBest(array(trainingSet), trainingLabels, 500)#训练权重
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is:%f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f" % (numTests, errorSum / float(numTests)))


# multiTest()
