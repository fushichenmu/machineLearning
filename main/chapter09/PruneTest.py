from chapter09.CartTest import *

'''
结果可见：
    相比于最小样本数，容差的预剪枝收敛速度更快
'''
def pre_prune_test_1():
    ex0 = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\ex0.txt", header=None, sep="\t")
    for i in range(2):
        for j in range(25):
            tree = createTree(ex0,ops = (i, j))
            print('-'*10,f'容差tolS={i},',f'最少样本数tolN={j}','-'*60)
            print(tree)

'''
结果可见：
    容差对数量级非常敏感
'''
def pre_prune_test_2():
    ex2 = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\ex2.txt", header=None, sep="\t")
    print(ex2.describe())
    # 探索tolS
    for i in np.arange(0, 5000, 500):
        ex2tree = createTree(ex2, ops=(i, 4))
        print(ex2tree)
        print('-' * 10, f'容差tolS={i}', '-' * 60)
    # 探索tolN
    for j in np.arange(0, 100, 10):
        ex2tree = createTree(ex2, ops=(3000, j))
        print('-' * 10, f'最少样本数tolN={j}', '-' * 60)
        print(ex2tree)
# pre_prune_test_2()



# print(train.iloc[:,0].count() ==train.iloc[:,1].count())
# plt.scatter(train.iloc[:,0].values,train.iloc[:,1].values)
# plt.scatter(test.iloc[:,0].values,test.iloc[:,1].values)
# plt.show()

"""
辅助函数1：
    判断是否是树类型
"""
def isTree(obj):
    return type(obj).__name__ == 'dict'

"""
辅助函数2：
    该函数是一个递归函数，从上到下遍历树直到叶节点为止，如果找到两个叶节点，则计算它们的平均
值。该函数对树进行了塌陷处理（即返回树平均值）
"""
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] =getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    mean = (tree['right'] + tree['left']) / 2.0
    return mean

"""
主函数：
    回归树剪枝函数
原理：
    基于已有的树切分测试数据：
        如果存在任意子集一棵树，则再该自己递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
"""
def prune(tree,testData):
    # 如果没有测试数据，则对树进行塌陷处理
    if testData.shape[0]==0:
        return getMean(tree)
    # 递归调用函数prune()对测试集进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    # 对左子树进行剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 对右子树进行剪枝
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 对叶节点进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge=sum(np.power(lSet.iloc[:,1]-tree['left'],2))+sum(np.power(rSet.iloc[:,1]-tree['right'],2))
        treeMean= (tree['right']+tree['left'])/2.0
        errorMerge= sum((testData.iloc[:,-1]-treeMean)**2)
        if errorMerge <errorNoMerge:
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        return tree

"""
后剪枝测试
"""
def pruneTest():
    ex2 = pd.read_csv("C:\\Users\\Mypc\\Desktop\\第8期 树回归（完整版）\\ex2.txt", header=None, sep="\t")
    train = ex2.iloc[:140,:]
    train.index = range(train.shape[0])
    test = ex2.iloc[140:,:]
    test.index = range(test.shape[0])
    tree = createTree(train,ops = (0, 1))
    prune(tree,test)
    print(tree)

# pruneTest()