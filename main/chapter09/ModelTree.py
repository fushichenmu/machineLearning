from chapter09.CartTest import *
from chapter09.PruneTest import *
"""
函数功能：计算特征矩阵、标签矩阵、回归系数
参数说明：
dataSet：原始数据集
返回：
ws：回归系数
X：特征矩阵（第一列增加x0=1）
Y：标签矩阵
"""


def linearSolve(dataSet):
    m, n = dataSet.shape
    con = pd.DataFrame(np.ones((m, 1)))  # 补充常数项X0=1
    conX = pd.concat([con, dataSet.iloc[:, :-1]], axis=1, ignore_index=True)
    X = np.mat(conX)  # 数据项
    Y = np.mat(dataSet.iloc[:, -1].values).T  # 标签项
    xTx = X.T * X
    if (np.linalg.det(xTx) == 0):
        raise NameError('奇异矩阵无法求逆，请尝试增大tolN,即ops第二个值')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


"""
辅助函数1：
    生成模型树的叶节点（即线性方程），这里返回的是回归系数
"""


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


"""
辅助函数2：
    计算给定数据集的误差（误差平方和）
"""


def modelError(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    modelError = sum(np.power(Y - yHat, 2))
    return modelError




def modelTest():
    exp2 = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\exp2.txt", header=None, sep="\t")
    print(exp2.describe())
    plt.scatter(exp2.iloc[:, 0].values, exp2.iloc[:, 1].values)
    plt.show()

    '''注意，方法名一定要传对'''
    print(createTree(exp2,modelLeaf,modelError,(1, 10)))



#导入训练集
biketrain =  pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\bikeSpeedVsIq_train.txt", header=None, sep="\t")

# biketrain.head()
#探索训练集
# biketrain.shape
# biketrain.describe()
# plt.scatter(biketrain.iloc[:,0],biketrain.iloc[:,1])
#导入测试集
biketest = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\bikeSpeedVsIq_test.txt", header=None, sep="\t")
# biketest.head()
#探索测试集
# biketest.shape
# biketest.describe()
# plt.scatter(biketest.iloc[:,0],biketest.iloc[:,1])
# plt.show()

"""
决策树预测辅助函数1：
    回归树叶节点预测函数，由于回归树的叶节点中是均值，所以可以直接返回该值。为了与下面模型树的预
测函数保持一致，这里仍保留两个输入参数，虽然我们只用了一个参数。
"""
def regTreeEval(model,inData):
    return model

"""
函数功能：
    返回模型树的叶节点预测结果
参数说明：
    model：模型树的叶节点，即回归系数
    inData：不带标签列的单个测试数据
"""
def modelTreeEval(model,inData):
    n= len(inData)
    X = np.mat(np.ones((1,n+1))) #因为多了个常数项，故是n+1
    X[:,1:n+1] = inData
    return X*model

"""
函数功能：
    返回单个测试数据的预测结果
参数说明：
    tree：字典形式的树,已经训练好的
    inData：单条测试数据
    modelEval：叶节点预测函数
"""
def treeForeCast(tree,inData,modelEval =regTreeEval):
    # 先判断是不是叶节点，如果是叶节点直接返回预测结果
    if not isTree(tree):
        return modelEval(tree,inData)
    # 根据索引找到左右子树
    if inData[tree['spInd']] > tree['spVal']:
        # 如果左子树不是叶节点，则递归找到叶节点
        if  isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
"""
函数功能：
    返回整个测试集的预测结果
参数说明：
    tree:字典形式的树
    testData:测试集
    modelEval：叶节点预测函数
返回：
    yHat:每条数据的预测结果
"""
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = testData.shape[0]
    yHat= np.mat(np.zeros((m,1)))
    for i in range(m):
        inData = testData.iloc[i,:-1].values
        yHat[i,0] = treeForeCast(tree,inData,modelEval)
    return yHat

def simpleRegressionOfCart():
    #创建回归树
    regTree = createTree(biketrain,ops=(1,20))
    #回归树预测结果
    yHat = createForeCast(regTree,biketest, regTreeEval)
    #计算相关系数R2
    print(np.corrcoef(yHat.T,biketest.iloc[:,-1].values)[0,1])
    #计算均方误差SSE
    print(sum((yHat.A.flatten()-biketest.iloc[:,-1].values)**2))

def simpleRegressionOfCartMaxCorrcoef():
    tolS = []
    tolN = []
    R2 = []
    SSE = []
    for i in range(5):
         for j in np.arange(1,100,10):
            regtree = createTree(biketrain,ops=(i,j))
            yHat = createForeCast(regtree,biketest,regTreeEval)
            r2 = np.corrcoef(yHat.T,biketest.iloc[:,-1].values)[0,1]  # corrcoef必须是要行向量
            sse = sum((yHat.A.flatten()-biketest.iloc[:,-1].values)**2) # .A转换为ndarray
            tolS.append(i)
            tolN.append(j)
            R2.append(r2)
            SSE.append(sse)
    df = pd.DataFrame([tolS,tolN,R2,SSE],index=['tolS','tolN','R2','SSE']).T
    print(df.loc[df['R2']==df['R2'].max(),:])
def modelTreeRegressionMaxCorrcoef():
    tolS_1 = []
    tolN_1 = []
    R2_1 = []
    SSE_1 = []
    for i in range(5):
        # 此处j≤16，则矩阵为奇异矩阵
        for j in np.arange(20, 100, 10):
            modeltree = createTree(biketrain, modelLeaf, modelError, ops=(i, j))
            yHat_1 = createForeCast(modeltree, biketest, modelTreeEval)
            r2_1 = np.corrcoef(yHat_1.T, biketest.iloc[:, -1].values)[0, 1]
            sse_1 = sum((yHat_1.A.flatten() - biketest.iloc[:, -1].values) ** 2)
            tolS_1.append(i)
            tolN_1.append(j)
            R2_1.append(r2_1)
            SSE_1.append(sse_1)
    df1 = pd.DataFrame([tolS_1,tolN_1,R2_1,SSE_1],index=['tolS','tolN','R2','SSE']).T
    df1.head()
    #找到最大相关系数R2和最小的SSE
    print(df1.loc[df1['R2']==df1['R2'].max(),:][0])

def standardRegressionMaxCorrcoef():
    #标准线性回归
    ws,X,Y = linearSolve(biketrain)
    ws
    #在第一列增加常数项1,构建特征矩阵
    testX = pd.concat([pd.DataFrame(np.ones((biketest.shape[0],1))),biketest.iloc[:,:-1]],
        axis=1,ignore_index = True)
    testMat = np.mat(testX)
    #标准线性回归预测结果
    yHat_2 = testMat*ws
    #相关系数R2
    R2_2 = np.corrcoef(yHat_2.T,biketest.iloc[:,-1].values)[0,1]
    #均方误差SSE
    SSE_2 = sum((yHat_2.A.flatten()-biketest.iloc[:,-1].values)**2)
