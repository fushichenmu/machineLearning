# 此类用于学习LASSO相关知识
# 由于直接求解LASSO非常复杂，一般使用其他方式求得近似解：LAR,向前逐步回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
"""
向前逐步回归伪代码：
数据标准化，使其分布满足0均值和单位方差
在每次迭代过程中：
    设置当前最小误差lowestError为正无穷
    对每个特征：
        增大或缩小：
            改变一个系数得到一个新的W
            计算新W下的误差
            如果误差Error小于当前最小误差lowestError:设置Wbest等于当前W
        将W设置为新的Wbest
"""
def stageWise(dataSet,eps=0.01,numIt = 100):
    xMat, yMat = get_Mat(dataSet)

    yMean = np.mean(yMat,0)                     #数据标准化
    yMat= yMat -yMean                           #数据标准化
    xMat = regularize(xMat)                     #数据标准化

    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))             #初始化迭代numIt次的回归系数矩阵
    ws = np.zeros((n,1))                        #初始化每次迭代的回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):                      #迭代numIt次
        lowestError = np.inf                    #初始化最小误差无正无穷
        for j in range(n):                      #对每个特征
            for sign in [-1,1]:                 #讨论增加和缩减的情况
                wsTest = ws.copy()
                wsTest[j] += eps*sign           #微调回归系数
                yHat = xMat * wsTest            #计算估计值
                sse = rssError(yMat,yHat)       #计算平方误差
                if(sse < lowestError):          #如果误差更小，则替换之
                    lowestError = sse
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T                   #记录每次迭代后的回归系数
    return returnMat

"""
样本矩阵归一化
"""
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

"""
计算SSE
"""
def rssError(yMat,yHat):
    sse = ((yMat.A- yHat.A)**2).sum()
    return sse

"""
导入数据集
"""
def get_data(path):
    dataSet =pd.read_table(path,header=None)
    dataSet.columns = ['性别','长度','直径','高度','整体重量','肉重量','内脏重量','壳重','年龄']
    # print(dataSet.shape)
    # print(dataSet.head(10))
    # print(dataSet.info())
    return dataSet

"""
函数说明：
    构建辅助函数  
输入参数：
    dataSet: DF数据集（最后一列为标签）
返回：
    特征矩阵和标签矩阵
"""
def get_Mat(dataSet):
    xMat = np.mat(dataSet.iloc[:,:-1].values) # xMat此时就是 X的T
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat


"""
函数说明：修正版最小二乘法
"""
def standRegres0(dataSet):
    xMat, yMat = get_Mat(dataSet)
    yMean = np.mean(yMat,0)
    yMat = yMat -yMean
    xMat = regularize(xMat)
    xTx =(xMat.T*xMat)
    if(np.linalg.det(xTx)==0):
        print("奇异矩阵，不可求逆")
        return
    ws = xTx.I *(xMat.T*yMat)
    return ws


# dataSet = get_data('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\abalone.txt')
# wsMat =stageWise(dataSet,eps=0.01,numIt=200)
# print(wsMat)
# ws = standRegres0(dataSet)
# wsMat2 =stageWise(dataSet,eps=0.001,numIt=5000)
#
# print(wsMat2)
# print(ws.T[0])
# plt.plot(wsMat)
# plt.xlabel("迭代次数")
# plt.ylabel("回归系数")
# plt.show()

