from chapter11.CorrelationRuleTest import *

raw_data = pd.read_csv(r'C:\Users\Mypc\Desktop\第10期 Apriori算法\agaricus-lepiota.data.txt', header=None)
raw_data.columns = ['蘑菇类型', '帽形状', '帽表面', '帽颜色', '有无挫伤', '气味', '菌褶附属物', '菌褶间距', '菌褶尺寸', '菌褶颜色', '茎形状', '茎根部', '蕈圈上部茎表面',
                    '蕈圈下部茎表面', '蕈圈上部茎颜色', '蕈圈下部茎颜色', '网类型', '网颜色', '蕈圈数量', '蕈圈类型', '孢子颜色', '数量', '栖息地']
#修改数据
for i in range(raw_data.shape[1]):
    length = len(raw_data.iloc[:,i].value_counts())
    if(length <=1):
        print(raw_data.columns[i])

raw_data.drop('网类型',axis=1,inplace=True)

fix_data = raw_data.copy()
for i in range(1,fix_data.shape[1]):
    fix_data.iloc[:, i] = fix_data.iloc[:, i].apply(lambda x: fix_data.columns[i] + '-' + x)

#将DF变为列表
mlist = fix_data.values.tolist()


L,suppData= apriori(mlist, minSupport=0.3)
#查看一共挖掘出了多少条频繁项集
n=0
for i in range(len(L)):
    n+=len(L[i])
print(n)
#查看这些频繁项集中有哪些是与毒蘑菇相关的
L0 = []
for i in range(len(L)):
    for item in L[i]:
        if item.intersection('p'):
            L0.append(item)
print(L0)


#执行度大于95%的关联规则
rules= generateRules(L,suppData, minConf=0.95)

len(rules) #共有4332条满足要求的规则
#从所有满足要求的规则中找出有毒蘑菇有关的规则
for i in range(len(rules)):
    if frozenset({'p'}) in rules[i]:
        print(rules[i])