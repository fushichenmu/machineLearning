import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
# % matplotlib inline

# 获取原数据
sampleSet = pd.read_csv('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\ex0.txt', header=None, sep='\t')

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

xMat,yMat = get_Mat(sampleSet)
x = 0.5
xi = np.arange(0,1.0,0.01)
k1,k2,k3 = 0.5,0.1,0.01
w1 = np.exp((xi-x)**2/(-2*k1**2)) # 注意分母带括号！！
w2 = np.exp((xi-x)**2/(-2*k2**2))
w3 = np.exp((xi-x)**2/(-2*k3**2))
#创建画布
fig =plt.figure(figsize=(6,8),dpi=100)
#子画布1，原始数据集
fig1 = fig.add_subplot(411)
plt.scatter(xMat.A[:,1],yMat.A,c="b",s=5)
#子画布2 ,W =0.5
fig2 = fig.add_subplot(412)
plt.plot(xi,w1,color='r')
plt.legend(['k = 0.5'])
#子画布3 ,W =0.1
fig3 = fig.add_subplot(413)
plt.plot(xi,w2,color='g')
plt.legend(['k = 0.1'])
#子画布4 ,W =0.01
fig4 = fig.add_subplot(414)
plt.plot(xi,w3,color='orange')
plt.legend(['k = 0.01'])
#show
plt.show()


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

def LWLR(testMat,xMat,yMat,k=1.0):
    m= np.shape(xMat)[0]
    n = np.shape(testMat)[0]
    weights = np.mat(np.eye(m)) # 每个训练数据都有对应权重，每个权重都是测试数据与所有训练数据的学习结果
    yHat =np.zeros(n)
    for i in range(n):
        for j in range(m):
            diffMat = testMat[i] - xMat[j]
            weights[j,j] = np.exp(diffMat*diffMat.T/(-2*k**2))
        xTx = xMat.T *(weights*xMat)
        if( np.linalg.det(xTx) == 0 ):
            print('矩阵非满秩矩阵，不能求逆')
            return
        ws = xTx.I *(xMat.T* (weights* yMat))  # 每一条测试数据计算的权重都不一样，虽然挺高了精度，但是计算量会特别大
        yHat[i] = testMat[i]*ws
    return yHat

sortIndexes = xMat[:,1].argsort(0) #升序 变成np.array了！
xSort = xMat[sortIndexes][:,0] # 对于任意一行，取第0列，因为有坑，外头多了一层
#计算不同k取值下的yHat
yHat1 = LWLR(xMat,xMat,yMat,k=1.0)
yHat2 = LWLR(xMat,xMat,yMat,k=0.1)
yHat3 = LWLR(xMat,xMat,yMat,k=0.003)
#创建画布
fig = plt.figure(figsize=(6,8),dpi=100)
#子图1绘制k=1.0的曲线
fig1 = fig.add_subplot(311)
plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5)
plt.plot(xSort[:,1],yHat1[sortIndexes],linewidth = 1, color='r')
plt.title('局部加权线性回归，k=1.0',size= 10,color='r')
#子图1绘制k=0.1的曲线
fig2 = fig.add_subplot(312)
plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5)
plt.plot(xSort[:,1],yHat2[sortIndexes],linewidth = 1, color='r')
plt.title('局部加权线性回归，k=0.1',size= 10,color='r')
#子图1绘制k=0.003的曲线
fig3 = fig.add_subplot(313)
plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5)
plt.plot(xSort[:,1],yHat3[sortIndexes],linewidth = 1, color='r')
plt.title('局部加权线性回归，k=0.003',size= 10,color='r')
#调整子图间距
plt.tight_layout(pad=1.2)
plt.show()

#相关系数比较
print(np.corrcoef(yMat.T,yHat1.T))
print(np.corrcoef(yMat.T,yHat2.T))
print(np.corrcoef(yMat.T,yHat3.T))