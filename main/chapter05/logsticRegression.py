from math import *
from numpy import *
# from chapter05.paintUtils import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch05\\testSet.txt", )
    arrayOLines = fr.readlines()
    for line in arrayOLines:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 这个是分类标签
    return dataMat, labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#普通梯度上升
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 将传入的数据集列表转化为矩阵
    labelMat = mat(classLabels).transpose()  # 将传入的标签列表转化为其转置矩阵 ，即：[1*n] --> [n*1]
    m, n = shape(dataMatrix)  # 获取矩阵的规格，这里是m行，n列
    alpha = 0.001  # 初始化步长
    maxCycles = 500  # 初始化迭代次数
    weights = ones((n, 1))  # 初始化权重，即回归系数，这里是个向量n*1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 注意，这里是矩阵相乘！！！[m*n] * [n*1] =[m*1] 为啥不把[1*1]放入其中？？？
        error = (labelMat - h)  #用于梯度上升
        weights = weights + alpha * dataMatrix.transpose() * error  # 迭代权重 ，transponse是求其转置矩阵 这里是 [n*m] * [m*1] = [n*1]
    return weights

#随机梯度上升算法
def gradAscentBest(dataMatIn,classLabels,numIter =150):
    m,n = shape(dataMatIn)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex= int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex]*weights))
            error = classLabels[randIndex] -h
            weights= weights + alpha * error * dataMatIn[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # weights=wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]#最佳拟合直线
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


# dataMat, labelMat = loadDataSet()
# weights = gradAscentBest(array(dataMat), labelMat,150)
# plotBestFit(weights)
# print(ones((5,1)))