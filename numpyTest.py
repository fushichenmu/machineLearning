from numpy import *
import operator
import  matplotlib
import matplotlib.pyplot as plt
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classify0(inX,dataSet,labels,k):#inX是测试向量，dataSet是样本向量，labels是样本标签向量，k用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0] #得到数组的行数。即知道有几个训练数据，0是行，1是列
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile将原来的一个数组，扩充成了dataSetSize个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat = diffMat**2 #各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1) #就是一行中的元素相加
    distances = sqDistances**0.5#开平方，以上是求出测试向量到样本向量每一行向量的距离
    sortedDistIndicies = distances.argsort()#对距离进行排序，从小到大
    classCount={}#构造一个字典，针对前k个近邻的标签进行分类计数。
    for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#得到距离最小的前k个点的分类标签
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)#对classCount字典分解为元素列表，使用itemgetter方法按照第二个元素的次序对元组进行排序，返回频率最高的元素标签，计数前k个标签的分类，返回频率最高的那个标签
    return sortedClassCount[0][0]


def file2matrix(filename):#将文本记录转换成numpy的解析程序
    fr = open(filename)#这是python打开文件的方式
    arrayOLines =fr.readlines()#自动将文件内容分析成一个行的列表
    numberOfLines = len(arrayOLines)#得到数据的个数
    returnMat = zeros((numberOfLines,3))#使用numpy下的zeros方法，返回的是一个被给定大小，数据类型和排列方式的填满0的数组
    classLableVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3]#前闭后开原则，只能去前三个数，存储到特征矩阵中
        classLableVector.append(int(listFromLine[-1]))#用-1表示最后一列元素，把标签放入这个向量中，这里必须明确的告诉解释器，存的是整形，否则当做字符串处理了
        index += 1
    return returnMat, classLableVector

#归一化处理 因为其中某一个变量的值，数字差值属性对计算的结果影响很大
def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue-minValue
    normDataSet = zeros(shape(dataSet))
    m =dataSet.shape[0]
    normDataSet = dataSet -tile(minValue,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minValue

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLables = file2matrix("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\datingTestSet2.txt")
    normDataSet, ranges, minValue = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyresult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLables[numTestVecs:m],3)
        print("the calssifier came back with: %d,the real answer is:%d", classifyresult, datingLables[i])
        if (classifyresult != datingLables[i]):
            errorCount +=1
    print("the total error rate is: %f" ,(errorCount / float(numTestVecs)) ) # 最后打印出测试错误率)

#输入某人的信息，便得出对对方喜欢程度的预测值
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))#输入
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\datingTestSet2.txt') #读入样本文件，其实不算是样本，是一个标准文件
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化
    inArr = array([ffMiles, percentTats, iceCream])#组成测试向量
#    pdb.set_trace()#可debug
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)#进行分类
#    return test_vec_g,normMat,datingLabels
    print('You will probably like this person:', resultList[classifierResult - 1])#打印结果


# returnMat, classLableVector =file2matrix("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch02\\datingTestSet2.txt")
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(returnMat[:,0], returnMat[:,1],15.0*array(classLableVector),15.0*array(classLableVector))
# plt.show()

# normDataSet,ranges,minValue = autoNorm(returnMat)
# datingClassTest()
# classifyPerson() u