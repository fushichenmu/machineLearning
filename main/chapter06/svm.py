# -*- coding:UTF-8 -*-
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import types

"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签

"""


def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat


"""
函数说明:随机选择alpha

Parameters:
    i - alpha
    m - alpha参数个数
"""


def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


"""
函数说明:修剪alpha

Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - alpah值
"""


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
函数说明:简化版SMO算法

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
"""


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 转换为numpy的mat存储
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    # 初始化b参数，统计dataMatrix的维度
    b = 0
    m, n = np.shape(dataMat)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter = 0
    # 最多迭代matIter次
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            fXi = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率。alpha值只考虑在（0，C）上的，即支持向量。因为如果alpha =0或C，后边修改就不会被修改了
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or (
                    (labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaI_old = alphas[i].copy()
                alphaJ_old = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if (L == H):
                    print("L==H");
                    continue
                # 步骤3：计算eta 如果eta>=0，alphaJ计算会很麻烦，实际不常出现这种情况，故continue
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[
                                                                                                                j, :].T
                if (eta >= 0):
                    print("eta>=0");
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                # abs:绝对值
                if (abs(alphas[j] - alphaJ_old) < 0.00001):
                    print("j not moving enough");
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJ_old - alphas[j])

                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaI_old) * dataMat[i, :] * dataMat[i, :].T - labelMat[j] * (
                        alphas[j] - alphaJ_old) * dataMat[i, :] * dataMat[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaI_old) * dataMat[i, :] * dataMat[j, :].T - labelMat[j] * (
                        alphas[j] - alphaJ_old) * dataMat[j, :] * dataMat[j, :].T

                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i] and (C > alphas[i])):
                    b = b1
                # 统计优化次数
                elif (0 < alphas[j] and (C > alphas[j])):
                    b = b2
                # 打印统计信息
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("item : %d i: %d,pairs changed %d" % (iter, i, alphaPairsChanged))
        # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

"""
函数说明:分类结果可视化

Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线解决
"""


# def showClassifer(dataMat, w, b):
#     # 绘制样本点
#     data_plus = []  # 正样本
#     data_minus = []  # 负样本
#     for i in range(len(dataMat)):
#         if   labelMat[i] > 0:
#             data_plus.append(dataMat[i])
#         else:
#             data_minus.append(dataMat[i])
#     data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
#     data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
#     plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
#     plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
#     # 绘制直线
#     x1 = max(dataMat)[0]
#     x2 = min(dataMat)[0]
#     a1, a2 = w
#     b = float(b)
#     a1 = float(a1[0])
#     a2 = float(a2[0])
#     y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
#     plt.plot([x1, x2], [y1, y2])
#     # 找出支持向量点
#     for i, alpha in enumerate(alphas):
#         if abs(alpha) > 0: #绝对值大于0，此时表示支持向量
#             x, y = dataMat[i]
#             plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
#     plt.show()


"""
函数说明:计算w

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
Returns:
    无
"""


def get_w(dataMat, labelMat, alphas):
    #列表转np的array 矩阵转np的array
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    #array.reshape(-1,1)把m*1 -->1*m，只不过是array的形式
    #array.tile用于复制数据，(1,2)表示所有数据在行方向复制1遍（等于没有复制），列方向复制2遍（翻了一倍）
    #array.dot求其内积
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

# 核函数，把低维数据转换为更高维数据
"""
X是所有数据 ,A是每一条数据， kTup是一个包含核函数信息的元祖
"""
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1))) # 初始化K矩阵:[m*1]
    if(kTup[0] == 'lin'): # 如果是线性核函数
        K = X * A.T # K矩阵就是X 和A的内积
    elif(kTup[0] == 'rbf'): # 如果是高斯核函数
        for j in range(m): #对于每一条数据
            deltaRow = X[j,:] - A #向量相减： x-y
            K[j] = deltaRow * deltaRow.T # ||x-y||^2
        K = np.exp(K/(-1*kTup[1]**2))  #kTup[1] = 根号2倍xita
    else: # 出错则报核函数构建失败
        raise  NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K #[m*1]


# if __name__ == '__main__':
#     dataMat, labelMat = loadDataSet('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch06\\testSet.txt')
#     b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
#     w = get_w(dataMat, labelMat, alphas)
#     showClassifer(dataMat, w, b)


# 完整的Platt SMO算法
class optStruct:

    """
    函数说明：初始化数据结构，维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 类别标签
        C - 惩罚系数
        toler - 容错率
    """
    def __init__(self, dataMatIn, classLabels, C, toler,kTup):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 类别标签
        self.C = C  # 惩罚系数
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m,self.m))) #初始化K矩阵[m*m] 为啥不是m*n?
        for i in range(self.m):
            self.K[:,i] = np.array(kernelTrans(self.X, self.X[i, :], kTup)) # 所有数据的第i列填充之：[m*1]
"""
计算误差
"""
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


"""
函数说明：内循环启发方式
Parameters：
    i - 标号为i的数据的索引值
    oS - 数据结构
    Ei - 标号为i的数据误差
Returns:
    j, maxK - 标号为j或maxK的数据的索引值
    Ej - 标号为j的数据误差
"""
def selectJ(i, oS, Ei):

    maxK = -1;
    maxDeltaE = 0;
    Ej = 0  # 初始化值
    oS.eCache[i] = [1, Ei]  # 首先将输入值Ei在缓存中设置为有效的
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i: continue  # 若k=i，结束本次循环，并开始下一次循环
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 初次循环时，采用随机选择alpha_j
        j = selectJrand(i, oS.m)  # 随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  # 计算Ej
    return j, Ej  # j,Ej

"""
函数说明：计算Ek,并更新误差缓存
Parameters：
    oS - 数据结构
    k - 标号为k的数据的索引值
Returns:
    无
"""
def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存

def innerL(i,oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if (L == H):
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

"""
函数说明：选择第一个alpha值的外循环
Parameters：
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 惩罚系数
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    oS.b - SMO算法计算的b
    oS.alphas - SMO算法计算的alphas
"""
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over alltestSetRBF2
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历不在边界0和C的alpha  即支持向量
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)  #使用内循环选择第二个alpha
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #遍历一次后改为非边界遍历
        elif (alphaPairsChanged == 0): entireSet = True #如果alpha没有更新,计算全样本遍历
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

"""
    函数说明：测试函数
    Parameters:
        k1 - 使用高斯核函数的时候表示到达率
    Returns:
        无
"""
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch06\\testSetRBF.txt')
    # 根据训练集计算b和alphas
    b, alphas = smoP(dataArr, labelArr, 500, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    # 利用nonzero函数获得非负alpha的索引值，进而得到支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    # 通过索引获得支持向量所对应的样本
    sVs = datMat[svInd]  # get matrix of only support vectors
    # 通过索引获得支持向量所对应的类别标签
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        # 计算各个点的核
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        # 根据支持向量的点，计算超平面，返回预测结果
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 加载测试集
    dataArr, labelArr = loadDataSet('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch06\\testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
    return [float(errorCount) / m,float(errorCount) / m]

def multiTest():
    numTests = 10; errorSum1=0.0;errorSum2=0.0
    for k in range(numTests):
        errorSum1+=float(testRbf()[0])
        errorSum2+=float(testRbf()[1])
    print ("在%d次迭代后训练集的错误率是: %.2f%%" % (numTests, errorSum1/float(numTests)))
    print ("在%d次迭代后测试集的错误率是: %.2f%%" % (numTests, errorSum2/float(numTests)))

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 20)):
    dataArr,labelArr = loadImages('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch06\\digits\\trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch06\\digits\\testDigits')
    errorCount = 0
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

testDigits(('rbf',20))
# testRbf(k1=20)
# dataMat = np.mat([(1, 2), (3, 4), (5, 6), (7, 8)])
# print(dataMat*dataMat[1,:].T)
