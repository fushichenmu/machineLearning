import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

"""
函数说明：
    计算数据误差率
入参：
    样本数据集
返回：
    样本误差率
"""


def calError(dataSet):
    m = np.shape(dataSet)[0]
    imax = dataSet.iloc[:, -1].value_counts()[0]
    error = 1 - imax / m
    return error


"""
函数说明：
    计算信息熵
入参：
    样本数据集
返回：
    样本信息熵
"""


def calEntropy(dataSet):
    m = np.shape(dataSet)[0]
    labels = dataSet.iloc[:, -1].value_counts()
    entropy = - (labels / m * (np.log2(labels / m))).sum()
    return entropy


"""
函数说明：
    计算基尼指数
入参：
    样本数据集
返回：
    样本基尼指数
"""


def calGini(dataSet):
    m = np.shape(dataSet)[0]
    labels = dataSet.iloc[:, -1].value_counts()
    p = labels / m
    gini = 1 - np.power(p, 2).sum()
    return gini


"""
函数说明：
    返回一个简易的数据集
返回：
    简易的数据集
"""


def createSimpleDataSet():
    samples = np.array([[1, 2], [1, 0], [2, 1], [0, 1], [0, 0]])
    labels = np.array([0, 1, 0, 1, 0])
    # 拼接函数，首先两者(可多者)转化为dataFrame形式，axis表示是按照列维度拼接，ignore_index表示忽略索引，即索引只管增加
    dataSet = pd.concat([pd.DataFrame(samples), pd.DataFrame(labels)], axis=1, ignore_index=True)
    return dataSet


"""
函数说明：
    根据特征值切分数据集
入参：
    dataSet:原始数据集
    feature:给定的特征列
    value:给定的特征值
返回：
    mat0：数据集1
    mat1:数据集2
注意：dataFrame格式切分后得到新的数据集，记住一定要更新对应的索引！
"""


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet.loc[dataSet.iloc[:, feature] > value, :]  # 注意都是中括号.loc函数是pandas里边索引数据的方式之一即loc[索引]
    mat0.index = range(mat0.shape[0])
    mat1 = dataSet.loc[dataSet.iloc[:, feature] <= value, :]
    mat1.index = range(mat1.shape[0])
    return mat0, mat1


"""
函数说明：
    计算总方差： 均方差* 样本数
入参：
    dataSet:原始数据集
返回：
    error:总方差
"""


def errType(dataSet):
    var = dataSet.iloc[:, -1].var() * dataSet.shape[0]  # 什么是均方差？？就是该列方差*样本总数
    return var


"""
函数说明：
    生成叶子节点，当我们的最佳分类函数确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。在回归树中
    该模型就是目标变量的均值
入参：
    dataSet:原始数据集
返回：
    leaf:叶子节点
"""


def leafType(dataSet):
    leaf = dataSet.iloc[:, -1].mean()
    return leaf

"""
函数说明：
   找到数据的最佳二元切分方式的函数
入参：
    dataSet:原始数据集
    leafType:生成叶子节点函数
    errType:生成总方差函数
    ops：用户定义的参数构成的元祖
返回：
    bestIndex:最佳特征索引
    bestValue:最佳特征切分值
"""
def chooseBestSplit(dataSet,leafType=leafType,errType=errType,ops=(1,4)):
    tolS =ops[0]; tolN = ops[1]                                                 #前者为允许的误差下降值，后者为切分的最小样本数
    if(len(set(dataSet.iloc[:,-1].values)) ==1):                                #如果当前所有值相等，则退出
        return None;leafType(dataSet)
    m,n = dataSet.shape                                                         #样本集合维度m,n
    S = errType(dataSet)                                                        #默认最后一个特征为最佳切分特征，计算其误差估计
    bestS= np.inf;  bestIndex = 0;  bestValue = 0                               #初始化最佳误差，最佳特征索引，最佳特征切分值
    for featIndex in range(0,n-1):                                              #遍历所有特征列,最后一列不考虑因为是结果
        colVal = set(dataSet.iloc[:, featIndex].values)                         #获取所有特征值并遍历
        for splitVal in colVal:
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)          #切分数据集
            if (mat0.shape[0] < tolN or mat1.shape[0] < tolN):                  #切分后如果样本数目不达标则跳过
                continue
            newS = errType(mat0)+errType(mat1)                                  #计算总方差，这里是两者之和
            if newS <bestS:                                                     #如果小于上一轮的bestS,说明当前feature和split不错
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS) < tolS:                                                       #循环完毕后，如果总方差下降的不大，说明此次挑选的不理想
        return None,leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)                 #再次获取，确保有数据,不排除极端情况，即循环体内每次都continue
    if(mat0.shape[0]<tolN or mat1.shape[0]<tolN):                               #如果得到样本数据小于tolN，表明不理想
        return None, leafType(dataSet)
    return bestIndex, bestValue

"""
函数说明：
   树构建函数
入参：
    dataSet:原始数据集
    leafType:生成叶子节点函数
    errType:生成总方差函数
    ops：用户定义的参数构成的元祖
返回：
    returnTree:构建的回归树
"""

def createTree(dataSet,leafType=leafType,errType=errType,ops=(1,4)):
    col,value =chooseBestSplit(dataSet,leafType,errType,ops)
    if col ==None :
        return value
    returnTree = {}
    returnTree['spInd'] = col
    returnTree['spVal'] = value
    leftSet,rightSet = binSplitDataSet(dataSet,col,value)
    returnTree['left'] = createTree(leftSet,leafType,errType,ops)
    returnTree['right'] = createTree(rightSet,leafType,errType,ops)
    return returnTree

"""
函数说明：
    用sklearn创建回归树
入参：
    dataSet：样本数据集
"""
def createTreeBySklearn(dataSet):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import linear_model
    x = (dataSet.iloc[:,1].values).reshape(-1,1)
    y = (dataSet.iloc[:,-1].values).reshape(-1,1)
    model1 = DecisionTreeRegressor(max_depth=1)
    model2 = DecisionTreeRegressor(max_depth=3)
    model3 = linear_model.LinearRegression()
    model1.fit(x, y)
    model2.fit(x, y)
    model3.fit(x, y)
    #预测
    X_test = np.arange(0,1,0.01)[:,np.newaxis]
    y_1 = model1.predict(X_test)
    y_2 = model2.predict(X_test)
    y_3 = model3.predict(X_test)
    #可视化结果
    plt.figure()
    plt.scatter(x,y,s=20,c='blue',label='data')
    plt.plot(X_test, y_1, color='cornflowerblue',label='max_depth1',linewidth=2)
    plt.plot(X_test, y_2, color='yellowgreen', label='max_depth3', linewidth=2)
    plt.plot(X_test, y_3, color='red', label='liner regression', linewidth=2)
    plt.xlabel("data")
    plt.ylabel("data")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

# dataSet = createSimpleDataSet()
# ex00 = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\ex0.txt", header=None, sep="\t")
# createTreeBySklearn(ex00)

# returnTree =createTree(ex00)
# print(returnTree)
# plt.scatter(ex00.iloc[:,0].values,ex00.iloc[:,1].values)
# plt.show()
# print(ex00.head())
