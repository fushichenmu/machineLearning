from chapter11.CorrelationRuleTest import *

votes = pd.read_csv(r'C:\Users\Mypc\Desktop\第10期 Apriori算法\house-votes-84.data.txt', sep=',', header=None)
votes.columns = ['党派', '残疾人婴幼儿提案', '水项目费用分摊', '预算决议案', '医生费用冻结决议案', '萨尔瓦多援助', '校园宗教团体决议', '反卫星禁试决议', '援助尼加拉瓜反政府',
                 'MX导弹决议案', '移民决议案', '合成燃料公司削减决议', '教育支出决议', '超级基金起诉权', '犯罪决议案', '免税出口决议案', '南非出口管理决议案']
#修改数据内容，将列名与数据合并
for i in range(1,votes.shape[1]):
    votes.iloc[:,i] = votes.iloc[:,i].apply(lambda x:x+'-'+votes.columns[i])
#将数据变为列表
voteslist=votes.values.tolist()

#获取频繁项集
L,suppData = apriori(voteslist, minSupport = 0.5)

#获取关联规则
rulesList = generateRules(L,suppData,0.9)

#支持度在50%，置信度在90%以上，得到的关联规则全是民主党的，这是为啥呢？
# party_counts = votes['党派'].value_counts() #统计各党派数量
# party_rate = party_counts/votes.shape[0] #计算各党派所占比例
# print(party_rate) # 因为democrat：0.613793 ；republican    0.386207<0.5!!!它根本就不会进入频繁项集

# print(votes.loc[votes['医生费用冻结决议案']=='y-医生费用冻结决议案']['党派'].value_counts())

'''
如果一定想要查看共和党的一些特征，挖掘与共和党相关的一些强关联规则，那么我们可以将支持度降低一点，
比如30%，然后再来进行频繁项集的挖掘、关联规则的提取。
但是需要提醒的是，如果降低了支持度，那频繁项集的数量一定会很大程度的增加，那么置信度的设定就要更
高一点，否则数据量大的话很容易卡死
'''
L,suppData = apriori(voteslist, minSupport = 0.3) #设定最小支持度为30%
brl = generateRules(L, suppData, minConf=0.99) #设定最小置信度为99%

#我们可以进一步探索，与民主党和共和党相关的强关联规则
for i in range(len(brl)):
    if frozenset({'republican'}) in brl[i] :
        print(brl[i])
    # if frozenset({'democrat'}) in brl[i]:
    #     print(brl[i])
