import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
函数说明：
    导入数据集
"""
def load_dataSet(path):
    dataSet = pd.read_csv(path,header=None,sep=',')
    return dataSet

"""
函数功能：计算两个数据集之间的欧式距离
输入：两个array数据集
返回：两个数据集之间的欧氏距离（此处用距离平方和代替距离）
"""
def distEclud(arrA,arrB):
    # return np.sum(np.power((arrA-arrB),2),axis=1)
    d = arrA - arrB
    dist = np.sum(np.power(d, 2), axis=1)
    return dist
"""
函数功能：
    随机生成k个质心
参数说明：
    dataSet:包含标签的数据集
    k：簇的个数
返回：
    data_cent：K个质心
"""
def randCent(dataSet,k):            #质心可以理解为一个点，这个点应该和任一样本数据有共同的维度数
    n = dataSet.shape[1]           #注意了，数据集最后一列是标签
    max_val =dataSet.iloc[:,:n-1].max()
    min_val = dataSet.iloc[:,:n-1].min()
    data_cent = np.random.uniform(min_val,max_val,(k,n-1))
    return data_cent

"""
函数功能：
    k-均值聚类算法
参数说明：
    dataSet：带标签数据集
    k:簇的个数
    distMeas:距离计算函数
    createCent:随机质心生成函数
返回：
    data_cent:质心
    result_set:所有数据划分结果
"""
def kMeans(dataSet,k,disMeas=distEclud,createCent=randCent):
    m, n = dataSet.shape
    data_cent = createCent(dataSet,k)
    clusterAssment = np.zeros((m,3))
    clusterAssment[:,0] = np.inf
    clusterAssment[:,1:3] = -1
    result_Set = pd.concat([dataSet,pd.DataFrame(clusterAssment)],axis=1,ignore_index=True)
    clusterChanged = True #初始化迭代标识
    while clusterChanged:
        clusterChanged = False
        #计算每个点到所有质心的欧式距离然后取最小值
        for i in range(m):
            dist = disMeas(dataSet.iloc[i,:n-1].values,data_cent)
            result_Set.iloc[i,n] = dist.min()
            result_Set.iloc[i, n+1] = np.where(dist == dist.min())[0]
        #更新迭代标识
        clusterChanged = not (result_Set.iloc[:,-1] == result_Set.iloc[:,-2]).all()
        #如果样本的簇存在改变了，那就必须更新所有质心以及上次质心索引
        if clusterChanged:
            cent_mean = result_Set.groupby(n+1).mean()
            data_cent = cent_mean.iloc[:,:n-1].values
            result_Set.iloc[:, -1]=result_Set.iloc[:, -2]
    return data_cent,result_Set
""" 
测试集验证
"""
def kMeansTest():
    dataSet = pd.read_csv(r'C:\Users\Mypc\Desktop\kmean\testSet.txt',header=None,sep='\t')
    print(dataSet.info())
    #因为测试集没有标签列，故我们认为制造一个标签列
    m=dataSet.shape[0]
    labels = pd.DataFrame(np.zeros((m,1)).reshape(-1,1))
    dataSet= pd.concat([dataSet,labels],axis=1,ignore_index=True)
    data_cent, result_Set = kMeans(dataSet,4)
    print(data_cent)
    plt.scatter(result_Set.iloc[:,0],result_Set.iloc[:,1],c=result_Set.iloc[:,-1])
    plt.scatter(data_cent[:,0],data_cent[:,1], color='red',marker='x',s=100)
    plt.show()

# kMeansTest()


"""
函数功能：聚类学习曲线
参数说明：
dataSet：原始数据集
cluster：Kmeans聚类方法
k：簇的个数
返回：误差平方和SSE
"""
def kcLearningCurve(dataSet,cluster=kMeans,k=10):
    n =dataSet.shape[1]
    SSE=[]
    for i in range(1,k):
        #取建议从2开始，质心为1的时候SSE数值过大，对后续曲线显示效果有较大影响
        data_cent,resultSet = cluster(dataSet,i+1)
        SSE.append(resultSet.iloc[:,n].sum())
    plt.plot(range(2, k+1), SSE, '--o')
    plt.show()
    return SSE


# irisData = pd.read_csv(r'C:\Users\Mypc\Desktop\kmean\iris.txt',header=None,sep=',')
'''在iris数据集中，质心选取3个或4个为佳，其中质心选取4个时的聚类效果比选取3个质心要更好。这里虽然生物
学上数据集对象本身包含了三个不同类别的鸢尾花，但就采集的数据和字段而言，从数据高维空间分布来说更加倾
向于可分为4个簇'''
# kcLearningCurve(irisData)
# testSet = pd.read_csv(r'C:\Users\Mypc\Desktop\kmean\testSet.txt',header=None,sep='\t')
# labels = pd.DataFrame(np.zeros((testSet.shape[0],1)).reshape(-1,1))
# testSet = pd.concat([testSet,labels],axis=1,ignore_index=True)
# kcLearningCurve(testSet)


