from math import *
from numpy import *


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


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))  # sigmoid函数

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 将传入的数据集列表转化为矩阵
    labelMat = mat(classLabels).transpose()  # 将传入的标签列表转化为其转置矩阵 ，即：[1*n] --> [n*1]
    m, n = shape(dataMatrix)  # 获取矩阵的规格，这里是m行，n列
    alpha = 0.001  # 初始化步长
    maxCycles = 500  # 初始化迭代次数
    weights = ones((n, 1))  # 初始化权重，即回归系数，这里是个向量n*1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 注意，这里是矩阵相乘！！！[m*n] * [n*1] =[m*1] 为啥不把[1*1]放入其中？？？
        error = (labelMat - h)  # 这个error是干嘛的？[n*1] -[m*1]  = ???
        weights = weights + alpha * dataMatrix.transpose() * error  # 迭代权重 ，transponse是求其转置矩阵 这里是 [n*m] * [m*1] = [n*1]
    return weights


dataMat, labelMat = loadDataSet()
print(gradAscent(dataMat, labelMat))
# print(ones((5,1)))
