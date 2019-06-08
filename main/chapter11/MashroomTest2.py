from chapter11.CorrelationRuleTest import *


raw_data = pd.read_csv(r'C:\Users\Mypc\Desktop\第10期 Apriori算法\agaricus-lepiota.data.txt', header=None)
raw_data.columns = ['蘑菇类型', '帽形状', '帽表面', '帽颜色', '有无挫伤', '气味', '菌褶附属物', '菌褶间距', '菌褶尺寸', '菌褶颜色', '茎形状', '茎根部', '蕈圈上部茎表面',
                    '蕈圈下部茎表面', '蕈圈上部茎颜色', '蕈圈下部茎颜色', '网类型', '网颜色', '蕈圈数量', '蕈圈类型', '孢子颜色', '数量', '栖息地']
raw_data.drop('网类型',axis=1,inplace=True)
fix_data = raw_data.copy()

#对数据集进行编码
ret =[] #放置所有列编码的结果（字典形式）
a=0
for i in range(fix_data.shape[1]):#对每一列进行编码操作
    dict = {}#每放置一列编码的结果
    col = fix_data.iloc[:,i].value_counts()#对每一列的值进行统计
    for ind,key in enumerate(col.index):#提取每一个值及其对应索引
        dict[key]=ind+a+1#编码从1开始，所以+1
    ret.append(dict)#将此列编码结果放置到re中
    for i in dict.values():a=i#当列编码的最后一个值

#把编码结果映射到数据集中
for i in range(len(ret)):
    fix_data.iloc[:,i] = fix_data.iloc[:,i].map(ret[i])

#这里需要注意的是，编码之后，数据集中所有的值均为int类型
#但是我们需要的是字符串类型，所以还需要进一步转换
for i in range(fix_data.shape[1]):
    fix_data.iloc[:,i] = fix_data.iloc[:,i].apply(lambda x:str(x))

mushdata = fix_data.values.tolist()
L,suppData= apriori(mushdata, minSupport=0.3) #挖掘频繁项集

#查看一共挖掘出了多少条频繁项集
n=0
for i in range(len(L)):
    n+=len(L[i])
print(n)

#查看这些频繁项集中有哪些是与毒蘑菇相关的
L0 = []
for i in range(len(L)):
    for item in L[i]:
        if item.intersection('2'):
            L0.append(item)
print(L0)

#创建毒蘑菇特征集合
re = []
for i in range(len(L0)):
    for item in L0[i]:
        if item not in re:
            re.append(item)
print(re)

#还原这些数字代表的原意义
ssc = {}
for i in range(fix_data.shape[0]):
    for j in range(fix_data.shape[1]):
        if fix_data.iloc[i,j] in re[1:]:
            ssc[raw_data.columns[j]]=raw_data.iloc[i,j]

#挖掘关联规则
rules= generateRules(L,suppData, minConf=0.95)
#提取与毒蘑菇有关的规则
for i in range(len(rules)):
    if frozenset({'2'}) in rules[i]:
        print(rules[i])