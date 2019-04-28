import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
观察样本数据
每条数据有三列：第一列为补充数据x0都是1.0，第二列为x1,第三列为y
"""
sampleSet = pd.read_csv('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\ex0.txt', header=None, sep='\t')
# print(sampleSet.shape)
# print(sampleSet.head(10))

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
函数功能：
    数据集可视化
"""
def plotShow(dataSet):
    xMat, yMat = get_Mat(dataSet)
    plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5) #注意这个地方需要用array,matrix是不行的；c是color;s是size
    plt.show()
# plotShow(sampleSet)
"""
核心函数功能：
    计算回归系数
参数说明：
    dataSet:原始数据集
返回：
    ws:回归系数(行向量)
"""
def standRegres(xMat, yMat):
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0:
        print('矩阵为奇异矩阵，无法求逆')
        return
    wx = xTx.I*(xMat.T*yMat)
    # print(wx)
    return wx

"""
绘制回归曲线
"""
def plotReg(dataSet):
    xMat, yMat = get_Mat(dataSet)
    plt.scatter(xMat.A[:,1],yMat.A,c = 'b' ,s =5)
    ws = standRegres(xMat, yMat)
    yHat = xMat * ws
    print(calSimilarity(yHat,yMat))
    plt.plot(xMat[:,1],yHat,c = 'r')
    plt.show()

"""
函数说明：
    计算相关系数（预测值与真实值的相似度）
"""
def calSimilarity(yHat,yMat):
    return np.corrcoef(yHat.T,yMat.T) # 一定要保证两者都是行向量




plotReg(sampleSet)