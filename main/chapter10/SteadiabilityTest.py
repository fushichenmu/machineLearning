from chapter10.KmeansTest import *
# 本脚本用于讨论模型收敛稳定性的探讨

'''
前言：
在执行前面聚类算法的过程中，好像虽然初始质心是随机生成的，但最终分类结果均保持一致.
若初始化参数是随机设置若初始化参数是随机设置（如此处初始质心位置是随机生成的），但
最终模型在多次迭代之后稳定输出结果，则可认为最终模型收敛。

迭代的条件：
kMeans聚类算法中我们设置的收敛条件比较简单，是最近两次迭代各点归属簇的划分结果，若
结果不发生变化，则表明结果收敛，停止迭代。
但是，这种依靠梯度下降获得局部最优解的方式，不一定是全局最优解。即我们使用均值作为
质心带入迭代，最终依据收敛判别结果计算的最终结果一定是基于初始质心的局部最优结果，
但不一定是全局最优的结果。
因此其实刚才的结果并不稳定，最终分类和计算的结果会受到初始质心选取的影响。

目的：
验证初始质心选取最终如何影响K-means聚类结果。

结论：
1.初始质心的随机选取在kMeans中将最终影响聚类结果；
2.质心数量从某种程度上将影响这种随机影响的程度，如果质心数量的选取和数据的空间集中分
布情况越相类似，则初始质心的随机选取对聚类结果可能造成影响的概率越小，当然，这也和质
心随机生成的随机方法也高度相关

改进措施：
尽量降低初始化质心随机性对最后聚类结果造成影响的方案有以下几种：

1.在设始化质心随机生成的过程中，尽可能的让质心更加分散；
    我们在利用np.random.random进行[0,1)区间内取均匀分布抽样而不是进行正态分布
    抽样，就是为了能够尽可能的让初始质心分散。
2.人工制定初始质心
    即在观察数据集分布情况后，手工设置初始质心，此举也能降低随机性影响，但需要人为干预；
3.增量的更新质心
    可以在点到簇的每次指派之后，增量地更新质心，而不是在所有的点都指派到簇中之后才更新簇质心。注
    意，每步需要零次或两次簇质心更新，因为一个点或者转移到一个新的簇（两次更新），或者留在它的当前
    簇（零次更新）。使用增量更新策略确保不会产生空簇，因为所有的簇都从单个点开始；并且如果一个簇只
    有单个点，则该点总是被重新指派到相同的簇。
'''

np.random.seed(123)
testSet = pd.read_csv(r'C:\Users\Mypc\Desktop\kmean\testSet.txt',header=None,sep='\t')
labels = pd.DataFrame(np.zeros((testSet.shape[0],1)).reshape(-1,1))
testSet = pd.concat([testSet,labels],axis=1,ignore_index=True)
'''
测试1:
    固定随机种子，k为3的聚类情况 
    这里我们每执行一次kMeans函数，初始质心就会随机生成一次
'''
def test1():
    for i in range(1,5):
        plt.subplot(2,2,i)
        test_cent,test_cluster =kMeans(testSet,3)
        plt.scatter(test_cluster.iloc[:,0].values,test_cluster.iloc[:,1].values,c=test_cluster.iloc[:,-2])
        plt.plot(test_cent[:,0],test_cent[:,1],'o',color='red')
    plt.show()

'''
测试2:
    固定随机种子，k为4的聚类情况 
    这里我们每执行一次kMeans函数，初始质心就会随机生成一次
'''
def test2():
    for i in range(1,5):
        plt.subplot(2,2,i)
        test_cent,test_cluster =kMeans(testSet,4)
        plt.scatter(test_cluster.iloc[:,0].values,test_cluster.iloc[:,1].values,c=test_cluster.iloc[:,-2])
        plt.plot(test_cent[:,0],test_cent[:,1],'o',color='red')
    plt.show()

'''
测试2:
    固定随机种子，k为5的聚类情况 
    这里我们每执行一次kMeans函数，初始质心就会随机生成一次
'''
def test3():
    for i in range(1,5):
        plt.subplot(2,2,i)
        test_cent,test_cluster =kMeans(testSet,5)
        plt.scatter(test_cluster.iloc[:,0].values,test_cluster.iloc[:,1].values,c=test_cluster.iloc[:,-2])
        plt.plot(test_cent[:,0],test_cent[:,1],'o',color='red')
    plt.show()

# test3()