import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
# % matplotlib inline

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
函数功能：
    计算LWLR的回归系数
入参：
    testMat:测试集
    xMat:训练集的特征矩阵
    yMat:训练集的标签矩阵
返回:
    yHat:函数预测值
评价：
    每来一条测试数据，就得循环所有的训练数据，不好
"""
def LWLR(testMat, xMat, yMat, k=1.0):
    m = np.shape(xMat)[0]
    n = np.shape(testMat)[0]
    weights = np.mat(np.eye(m))  # 每个训练数据都有对应权重，每个权重都是测试数据与所有训练数据的学习结果
    yHat = np.zeros(n)
    for i in range(n):
        for j in range(m):
            diffMat = testMat[i] - xMat[j]
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        if (np.linalg.det(xTx) == 0):
            print('矩阵非满秩矩阵，不能求逆')
            return
        ws = xTx.I * (xMat.T * (weights * yMat))  # 每一条测试数据计算的权重都不一样，虽然挺高了精度，但是计算量会特别大
        yHat[i] = testMat[i] * ws
    return yHat

"""
函数功能：
    切分训练集
参数说明：
    dataSet:原始数据集
    rate:切分比例
返回:
    train,test:切分好的训练集和测试集
"""
def randSplit(dataSet,rate):
    m = np.shape(dataSet)[0]
    n = int(m *rate)
    train= dataSet.iloc[:n,:]
    test = dataSet.iloc[n:m,:]
    test.index= range(test.shape[0]) #打乱数据
    return train,test
"""
函数说明：
    计算误差平方和SSE
参数说明：
    yMat:真实值
    yHat:预测值
返回：
    SSE：误差平方和
"""
def sseCal(yMat,yHat):
    SSE =((yMat.A.flatten()-yHat)**2).sum()
    return SSE

#训练并验证
def test():
    dataSet = get_data('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\abalone.txt')
    # train,test =  randSplit(dataSet,0.8)
    trainXMat,trainYMat=get_Mat(dataSet) #数据量太大
    # testXmat,testYmat = get_Mat(test)
    train_see =[]
    test_see=[]
    for k in np.arange(0.5, 10.1, 0.1): #这个不是3个值！！！numpy.arange(start, stop, step, dtype)
        trainyHat = LWLR(trainXMat[:99], trainXMat[:99], trainYMat[:99], k)
        sse1 = sseCal(trainYMat[:99],trainyHat)
        train_see.append(sse1)

        testYHat = LWLR(trainXMat[100:199], trainXMat[:99], trainYMat[:99], k)
        sse2 = sseCal(trainYMat[100:199],testYHat)
        test_see.append(sse2)
        a = np.array([0.5, 10.1, 0.1])
    plt.plot(np.arange(0.5, 10.1, 0.1),train_see,color='b')
    plt.plot(np.arange(0.5, 10.1, 0.1), test_see, color='r')
    plt.xlabel('不同的K值')
    plt.ylabel('SSE')
    plt.legend(['train_sse','test_sse'])
    plt.show()
#光100条数据都跑了2min
test()