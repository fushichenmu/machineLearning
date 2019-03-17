from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'maybe'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    lables = ['no surfacing', 'flippers']
    return dataSet, lables

# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    lableCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] += 1
    shannonEnt = 0.0
    for key in lableCounts:
        prop = lableCounts[key] / numEntries
        shannonEnt -= prop * log(prop, 2)
    return shannonEnt


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for item in dataSet:
        if item[axis] == value:
            reduceItem = item[:axis]
            reduceItem.extend(item[axis + 1:])
            retDataSet.append(reduceItem)
    return retDataSet


# 挑选出最优特征
def chooseBestFeatureToSplit(dataSet):
    fatherGain = calcShannonEnt(dataSet)
    numberOfFeature = len(dataSet[0]) - 1
    bestGain = 0.0
    bestFeature = -1
    for i in range(numberOfFeature):
        featDataList = [example[i] for example in dataSet]
        uniqueValues = set(featDataList)
        newGain = 0.0
        for value in uniqueValues:
            splitedDataSet = splitDataSet(dataSet, i, value)
            prop = len(splitedDataSet) / float(len(featDataList))
            newGain += prop * calcShannonEnt(splitedDataSet)
        infoGain = fatherGain - newGain
        if (infoGain > bestGain):
            bestGain = infoGain
            bestFeature = i
    return bestFeature


# 创建决策树
def createTree(dataSet, lables):
    classList = [example[-1] for example in dataSet]
    if (classList.count(classList[0]) == len(classList)):
        return classList[0]
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = lables[bestFeat]
    myTree = {bestFeatLable: {}}
    del (lables[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLables = lables[:]
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables)
    return myTree


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if (vote not in classCount.keys()):
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)  # 对classCount字典分解为元素列表，使用itemgetter方法按照第二个元素的次序对元组进行排序，返回频率最高的元素标签，计数前k个标签的分类，返回频率最高的那个标签
    return sortedClassCount[0][0]





# dataSet,lables = createDataSet()
# print(createTree(dataSet,lables))

