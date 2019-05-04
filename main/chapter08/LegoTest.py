from chapter08.lassoTest import *
from bs4 import BeautifulSoup

"""
函数说明:抓取一个网页的信息
参数说明:
    data:用来盛放抓取的所有信息
    inflie:html文件名
    yr:年份
    numPce:部件数目
    origPrc:出产价格
"""


def scrapePage(data, inflie, yr, numPce, origPrc):
    HTML_DOC = open(inflie, encoding='UTF-8').read()
    soup = BeautifulSoup(HTML_DOC, 'lxml')
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r=f'{i}')
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r=f'{i}')
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1):   #find函数用于搜索字符串在指定字符串的索引位置，如果没发现就返回负一
            newFlag = 1
        else:
            newFlag = 0
        # 查找是否已经标志出售，只收集已售的数据
        soldbutt = currentRow[0].find_all('td')[3].find_all('span')  #currentRow是个集合，故取【0】
        if (len(soldbutt) == 0):
            print(f"商品 #{i} 没有出售")
        else:
            # 解析页面获取当前价格
            priceStr = currentRow[0].find_all('td')[4]
            priceStr = priceStr.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if (len(priceStr) > 1):
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的价格
            if sellingPrice > origPrc * 0.5:
                data.append([yr, numPce, newFlag, origPrc, sellingPrice])
        i += 1
        currentRow = soup.find_all('table', r=f'{i}')


"""
爬取6个网页的数据
"""


def setDataCollect(data):
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego8288.html", 2006, 800,
               49.99)
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego10030.html", 2002, 3096,
               269.99)
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego10179.html", 2007, 5195,
               499.99)
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego10181.html", 2007, 3428,
               199.99)
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego10189.html", 2008, 5922,
               299.99)
    scrapePage(data, "C:\\Users\\Mypc\\Desktop\\machinelearninginaction\\Ch08\\setHtml\\lego10196.html", 2009, 3263,
               249.99)

"""
核心函数功能：
    计算回归系数
参数说明：
    dataSet:原始数据集
返回：
    ws:回归系数(行向量)
"""
def standRegres(xMat, yMat):
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0:
        print('矩阵为奇异矩阵，无法求逆')
        return
    wx = xTx.I*(xMat.T*yMat)
    # print(wx)
    return wx


data = []
setDataCollect(data)
#把数据变成dataFrame形式
df = pd.DataFrame(data)
df.columns =['出品年份','部件数目','是否全新','原价','售价']
#观察数据
# print(df.info())
# print(df.describe())
#在第0列增加常数项特征X0 =1      到底什么时候需要加入默认截距项？？？
col_name = df.columns.tolist()
col_name.insert(0,'x0')
df = df.reindex(columns = col_name)
df['x0'] = 1
print(df.head())

#用简单线性回归计算权重
xMat,yMat =get_Mat(df)
ws = standRegres(xMat,yMat)    #数据标准化有问题，这里用的是原始的线性回归
yHat = xMat * ws
#画出真实值和预测值的散点图
plt.scatter(range(df.shape[0]),yMat.A)
plt.scatter(range(df.shape[0]),yHat.A)
plt.show()

#计算相关性
print(np.corrcoef(yHat.T,yMat.T))