from chapter10.KmeansTest import *
# 此脚本用于研究二分K-均值算法
'''
二分K-均值算法伪代码：
    初始将所有点看做是一个簇
    当簇数小于K：
        对于每一个簇：
            计算总误差
            在给定的簇下进行2均值聚类
            计算划分后的2个新簇的总误差
        选择总误差更小的那个簇进行划分
'''

# 1.准备数据
testSet = pd.read_table(r'C:\Users\Mypc\Desktop\kmean\testSet.txt', header=None)
label = pd.DataFrame(np.zeros(testSet.shape[0]).reshape(-1, 1))
testSet = pd.concat([testSet, label], axis=1, ignore_index = True)

# 2.初始化


# 3.构建辅助函数
"""
函数功能：
    在给定质心的情况下划分各点所属簇
参数说明：
    dataSet:原始数据集
    centroids:质心
    distMeas:距离衡量函数
返回：
    result_set：划分结果
"""
def kMeans_assment(dataSet,centroids,distMeas=distEclud):
    m,n= dataSet.shape
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf
    clusterAssment[:, 1: 3] = -1
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1,ignore_index=True)
    for i in range(m):
        dist = distMeas(dataSet.iloc[i,:n-1].values,centroids)
        result_set.iloc[i,n] = dist.min()
        result_set.iloc[i, n + 1] = np.where(dist == dist.min())[0]
        result_set.iloc[:, -1] = result_set.iloc[:, -2]
    return result_set

#4.主函数：二分K均值函数
def binKmeans(dataSet,k,distMeas=distEclud):
    m,n=dataSet.shape
    centroids, result_set = kMeans(dataSet, 2)      #初始化二分
    j=2
    while j<k:
        result_tmp = result_set.groupby(n + 1).sum()
        clusterAssment = pd.concat([pd.DataFrame(centroids), result_tmp.iloc[:, n]]
                                   ,axis=1, ignore_index=True)  #质心容器拼接每个质心的SSE
        # lowestSSE = clusterAssment.iloc[:, n - 1].sum()
        centList=[]            #定义盛放质心的容器
        sseTotle= np.array([])  #定义每次二分后新得的总SEE
        for i in clusterAssment.index:
            df_tmp= result_set.iloc[:,:n][result_set.iloc[:,-1] == i]  #切分该质心下的簇样本
            df_tmp.index = range(df_tmp.shape[0])       #重构索引
            cent,res = kMeans(df_tmp,2,distMeas)        #尝试对该簇进行二分
            centList.append(cent)                       #将此次二分获得的2个新簇质心保存
            sseSplit = res.iloc[:, n].sum()             #计算二分后，2个新簇的SEE
            sseNotSplit = result_set.iloc[:, n][result_set.iloc[:, -1] != i].sum() #计算未进行二分的其他簇的总SEE
            sseTotle = np.append(sseTotle,sseSplit+sseNotSplit) #将每次二分后，所有数据样本的总SEE保存
        min_index = np.where(sseTotle == sseTotle.min())[0][0]   #循环完毕后，挑出最小SEE的索引
        clusterAssment = clusterAssment.drop([min_index])   #删除要被二分的原簇质心
        centroids = np.vstack([clusterAssment.iloc[:,:n-1].values,centList[min_index]]) #拼接二分得到的新的2个质心
        result_set = kMeans_assment(dataSet, centroids)  #重构数据
        j=j+1
    return centroids,result_set

'''
轮廓系数的具体实现
'''
def silhouetteCoe(centroids,result):
    result_set = result.copy()
    m,n = result_set.shape
    nc = len(centroids)
    result_list = []
    for i in range(nc):
        result_set[n+i] = 0
        result_temp = result_set[result_set.iloc[:,n-1] == i]
        result_temp.index = range(result_temp.shape[0])
        result_list.append(result_temp)
    result_set["a"] = 0
    result_set["b"] = 0
    for i in range(m):
        l_temp = []
        for j in range(nc):
            result_set.iloc[i, n + j] = distEclud(result_set.iloc[i,:n-4].values,result_list[j].iloc[:,:n-4].values).mean()
            if result_set.iloc[i,n-2] == j:
                result_set.loc[i,"a"] = result_set.iloc[i,n+j]
            else:
                l_temp.append(result_set.iloc[i,n+j])
        result_set.loc[i, "b"] = np.array(l_temp).min()
    result_set["s"] = (result_set.loc[:, "b"] - result_set.loc[:, "a"]) /(result_set.loc[:, "a":"b"].max(axis=1))
    return result_set["s"].mean()

# def silhouetteCoe(result):
#     result_set = result.copy()
#     m, n = result_set.shape
#     nc = len(centroids)
#     for i in range(nc):
#         result_set[n+i]=0
#     result_list = []
#     for i in range(nc):
#         result_temp = result_set[result_set.iloc[:, n-1] == i]
#         result_temp.index = range(result_temp.shape[0])
#         result_list.append(result_temp)
#     for i in range(m):
#         for j in range(nc):
#             result_set.iloc[i,n+j]=distEclud(result_set.iloc[i, :n-4].values,result_list[j].iloc[:, :n-4].values).mean()
#     result_set["a"]=0
#     result_set["b"]=0
#     for i in range(m):
#         l_temp=[]
#         for j in range(nc):
#             if(result_set.iloc[i,n-1] == j):
#                 result_set.loc[i,"a"] = result_set.iloc[i, n+j]
#             else:
#                 l_temp.append(result_set.iloc[i, n+j])
#         result_set.loc[i,"b"] = np.array(l_temp).min()
#     result_set["s"] = (result_set.loc[:,"b"]-result_set.loc[:,"a"]) / result_set.loc[:,"a":"b"].max(axis=1)
#     return result_set["s"].mean()


# centroids,result_set = binKmeans(testSet,4,distEclud)
np.random.seed(1288)
sil = []
for i in range(1, 7):
    centroids, result_set = binKmeans(testSet, i + 1)
    sil.append(silhouetteCoe(centroids,result_set))
plt.plot(range(2, 8), sil, '--o')
plt.show()
# plt.scatter(result_set.iloc[:,0], result_set.iloc[:, 1], c=result_set.iloc[:,-1])
# plt.plot(centroids[:, 0], centroids[:, 1], 'o', color='red')
# plt.show()