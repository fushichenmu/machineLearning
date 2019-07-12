# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
from fuzzywuzzy import fuzz
from main.test.LevenstainDistance import simility
def cal_lcs_sim(first_str, second_str):
    len_vv = [[0] * 50] * 50  # 第一个50为列数，第二个50为行数
    #
    # first_str = unicode(first_str, "utf-8", errors='ignore')
    # second_str = unicode(second_str, "utf-8", errors='ignore')

    len_1 = len(first_str.strip())
    len_2 = len(second_str.strip())

    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if first_str[i - 1] == second_str[j - 1]:
                len_vv[i][j] = 1 + len_vv[i - 1][j - 1]
            else:
                len_vv[i][j] = max(len_vv[i - 1][j], len_vv[i][j - 1])

    return float(float(len_vv[len_1][len_2] * 2) / float(len_1 + len_2))  #两个字符串的相似度


def test():
    data = open(r'C:\Users\qwe\Desktop\lcs_input.data', mode='r', encoding='utf-8')
    for line in data:
        ss = line.strip().split('\t')
        if len(ss) != 2:
            continue
        first_str = ss[0].strip()
        second_str = ss[1].strip()

        sim_score = cal_lcs_sim(first_str, second_str)
        print('\t'.join([first_str, second_str, str(sim_score)]))


# test()
first_str = '中国男子篮球职业联赛'
second_str = '中男篮职业联赛'
sim_score = cal_lcs_sim(first_str, second_str)    #LCS
fuzz_score = fuzz.ratio(first_str,second_str)     #FUZZ模块
leven_score = simility(first_str,second_str) #Levenstance

print('\t'.join([first_str, second_str, str(sim_score)]))
print('\t'.join([first_str, second_str, str(fuzz_score)]))
print('\t'.join([first_str, second_str, str(leven_score)]))