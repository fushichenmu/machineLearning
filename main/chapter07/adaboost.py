import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#获取数据矩阵以及分类标签
def get_Mat(path):
    dataSet = pd.read_table(path,header=None)
    xMat = np.mat(dataSet.iloc[:,:-1].values) #对于每一行，获取最后一个数据之前的所有数据
    yMat = np.mat(dataSet.iloc[:,-1].values).T#对于每一行，获取最后一个数据
    return xMat,yMat

#构建数据可视化函数，并运行查看数据分布
def show_plot(xMat,yMat):
    x= np.array(xMat[:,0])
    y= np.array(xMat[:,1])
    label = np.array(yMat)
    plt.scatter(x,y,c=label)
    plt.title('单层决策树测试数据')
    plt.show()

"""
函数功能：单层决策树分类函数
参数说明：
    xMat:数据矩阵
    i:第i列，也就是第几个特征
    Q：阈值
    S：标志
返回：
    re:分类结果
"""
def classify0(xMat,i,Q,S):
    re = np.ones((xMat.shape[0],1)) #构造初始分类结果矩阵1
    if(S == 'lt'):                  #使用了np的广播机制
        re[xMat[:,i] <= Q] = -1     #如果小于阈值，则赋值为-1
    else:
        re[xMat[:,i] > Q] = -1      #如果大于阈值，则赋值为-1
    return re

"""
函数说明：找到数据集上最佳的单层决策树
参数说明：
    xMat:特征矩阵
    yMat:标签矩阵
    D：样本权重
返回：
    bestStump:最佳单层决策树信息
    minE:最小误差
    bestClasses:最佳分类结果
"""
def get_Stump(xMat,yMat,D):
    m,n = xMat.shape                                #原始数据的维度m,n
    steps= 10                                       #初始化步数
    bestStump ={}                                   #初始化用字典存储树桩信息
    bestClasses = np.mat(np.zeros((m,1)))           #初始化最佳分类结果为1
    minE = np.inf                                   #初始化最小误差
    for i in range(n):                              #遍历所有的列
        min = xMat[:,i].min()                       #获取每列的最大最小值
        max = xMat[:,i].max()
        stepSize = (max-min)/steps                  #计算步长
        for j in range(-1,int(steps)+1):            #遍历每一个步长，选择最合适的阈值，肯定要考虑超出范围的情况，所以要多遍历两次
            for S in ['lt','gt']:                   #遍历大于和小于的情况
                Q = (min +j*stepSize)               #得到阈值
                re = classify0(xMat,i,Q,S)          #得到分类结果
                err = np.mat(np.ones((m,1)))        #初始化误差矩阵
                err[re == yMat] =0                  #分类正确，赋值为0
                eca = D.T*err                       #计算加权错误率，即每个样本的预测结果乘以其权重
                # print(eca)
                if eca < minE:
                    minE = eca                      #迭代最小错误率
                    bestClasses = re.copy()         #储存最佳分类树
                    bestStump['特征值'] = i         #储存其他相关信息
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump,minE,bestClasses

"""
完整的adaboost算法的实现：
对每一次迭代：
    利用bestStump()找到最佳的单层决策树
    将单层决策树加入到单层决策数组
    计算分类器权重alpha
    更新样本权重向量D
    更新累计类别估计值
    如果错误率为0了或者迭代次数到了，终止
参数说明：
    xMat:原始数据矩阵
    yMat:标签矩阵
    maxC:最大迭代次数
返回：
    weekClass:弱分类器信息
    aggClass:类别估计值（其实就是更改了标签的估计值）
"""
def Ada_train(xMat,yMat,maxC):
    weekClass=[]                                                #初始化弱分类器集合
    m = np.shape(xMat)[0]                                       #训练数据个数m
    D = np.mat(np.ones((m,1))/m)                                #初始化权重D
    aggClass = np.mat(np.zeros((m,1)))                          #初始化类别估计值
    for i in range(maxC):                                       #每次迭代
        bestStump, error, bestClasses =get_Stump(xMat,yMat,D)   #获取最佳分类决策树及对应的错误率
        alpha = float(0.5*np.log((1-error)/max(error,1e-16)))   #计算该分类器的权重alpha
        bestStump['alpha'] = np.round(alpha,2)                  #alpha储存起来，保留两位小数
        weekClass.append(bestStump)                             #添加到weekClass
        expon = np.multiply(-1*alpha*yMat,bestClasses)          #计算指数 ，因为正确分类时ymat*bestClasses为1，否则为-1；正确分类带负号
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()                                           #更新样本权重向量D
        aggClass += alpha * bestClasses                         #每个样本的估计值累加一下，因为错误的权重会大些。
        aggErr = np.multiply(np.sign(aggClass) !=yMat,np.ones((m,1)))   #因为aggClass编程浮点数了，故用sign() ,然后正确分类为0，错误分类为1
        errRate = aggErr.sum()/m                                        #aggErr是一个数组，故用Sum()求和
        if errRate ==0:
            break
    return weekClass,aggClass

"""
构造基于adaboost的分类器
参数说明：
    data:待分类样例
    weekClass:训练好的分类器
返回：
    分类结果
"""
def AdaClassify(data,weekClass):
    dataMat = np.mat(data)
    m =np.shape(dataMat)[0]
    aggClass= np.mat(np.zeros((m,1)))
    for i in range(len(weekClass)):
        classEst = classify0(dataMat,weekClass[i]['特征值']
                                    ,weekClass[i]['阈值']
                                    ,weekClass[i]['标志'])
        aggClass += weekClass[i]['alpha']*classEst
    return np.sign(aggClass)

xMat,yMat=get_Mat('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch07\\simpdata.txt')
weekClass,aggClass= Ada_train(xMat,yMat,maxC=9)
result = AdaClassify(xMat,weekClass)
print(result)
# show_plot(xMat,yMat)

