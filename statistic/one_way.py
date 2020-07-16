# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Test by wangliang

import os
os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/')
import numpy as np
import pandas as pd
import itertools
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
import scipy,math
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd,MultiComparison



'''
准备数据
'''
data = pd.read_excel('one_v.xlsx',index = False)
level1 = data[data['method'] == 1]['score']
level2 = data[data['method'] == 2]['score']
level3 = data[data['method'] == 3]['score']
list_levels = [level1,level2,level3]
list_total = data['score']

'''
一、正态性检验
'''
alpha = 0.05
def check_normality(testData,alpha = 0.05):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(testData) < 50:
        normaltest_statistic,normaltest_p = stats.normaltest(testData) #https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.normaltest.html
        print(normaltest_statistic,normaltest_p)
        if normaltest_p < alpha:
            print('use normaltest')
            print('data are not normal distributed')
            return False
        else:
            print('use normaltest')
            print('data are normal distributed')
            return True
    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(testData) < 50:
        shapiro_statistic,shapiro_p  = stats.shapiro(testData) #Perform the Shapiro-Wilk test for normality. https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.shapiro.html
        print(shapiro_statistic,shapiro_p)
        if shapiro_p < alpha:
            print("use shapiro:")
            print("data are not normal distributed")
            return False
        else:
            print("use shapiro:")
            print("data are normal distributed")
            return True
    if 300 >= len(testData) >= 50:
        lilliefors_statistic,lilliefors_p = lilliefors(testData) #https://blog.csdn.net/qq_20207459/article/details/103000285
        print(lilliefors_statistic,lilliefors_p)
        if lilliefors_p < alpha:
            print("use lillifors:")
            print("data are not normal distributed")
            return False
        else:
            print("use lillifors:")
            print("data are normal distributed")
            return True
    if len(testData) > 300:
        kstest_statistic,kstest_p = scipy.stats.kstest(testData, 'norm')
        print(kstest_statistic,kstest_p)
        if kstest_p < alpha:
            print("use kstest:")
            print("data are not normal distributed")
            return False
        else:
            print("use kstest:")
            print("data are normal distributed")
            return True

#  对所有样本组进行正态性检验
# 先将各个样本分好，对所有样本检验正态性，也是对样本组里的每个样本检验
def NormalTest(list_groups,alpha):
    for group in list_groups:
        # 正态性检验
        status = check_normality(group,alpha)
        if status == False:
            return False


'''
二、方差齐性检验   
'''

def Levene_test(*args,alpha = 0.05):
    leveneTest_statistic,leveneTest_p=scipy.stats.levene(*args)
    print(leveneTest_statistic,leveneTest_p)
    if leveneTest_p < alpha:
        print("variances of groups are not equal")
        return False
    else:
        print("variances of groups are equal")
        return True


'''
三、F检验/ANOVA 表
'''

# 单因素方差分析
model = ols('score ~ C(method)',data).fit()
anova_result = anova_lm(model)


#多因素方差分析
# formula = 'y~x1+x2+x1:x2'  # 公式 因变量~ 自变量1+自变量2+ 自变量1和2的交互效应
# model = ols(formula, data).fit()
# anova_results = anova_lm(model)
# print(anova_results)

'''
四、多重比较
'''

# 组内误差，也叫随机误差
def SE(level):
    se = 0
    mean1 = np.mean(level)
    for i in level:
        error = i-mean1
        se += error**2
    return se

# 组内误差平方和，也叫随机误差，是各个水平的误差平方和相加
def SSE(list_levels):
    sse = 0
    for every_level in list_levels:
        se = SE(every_level)
        sse += se
    return sse

# 组内误差均方
def MSE(list_levels,list_total):
    sse = SSE(list_levels)
    mse = sse/(len(list_total) - 1*len(list_levels))*1.0
    return mse


# 排列组合函数
def Combination(list_levels):
    combination= []
    for i in range(1,len(list_levels)+1):
        iter = combinations(list_levels,i)
        combination.append(list(iter))
    #需要排除第一个和最后一个
    return combination[1:-1][0]
# 两两比较
def LSD(list_levels,list_total,sample1,sample2): # p值怎么算
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    distance = abs(mean1-mean2)
    print('distance:::',distance)
    # t检验自由度
    df = len(list_total) - 1*len(list_levels)
    mse = MSE(list_levels,list_total)
    print('MSE:::',mse)
    t_half = stats.t(df).isf(alpha/2) # 由自由度计算T值
    print('t_half:::',t_half)
    lsd = t_half*math.sqrt(mse*(1.0/len(sample1)+1.0/len(sample2)))
    print('LSD:::',lsd)
    if distance<lsd:  # 是根据lsd值判断是否显著
        print('There is no significant difference between：',sample1,sample2) # 优化
    else:
        print('There is significant difference between:',sample1,sample2) # 优化
# 多重比较
def Multiple_test(list_levels):
    print('---------------------multiple test------------------')
    list_total = data.iloc[:,-1] # 这里需要注意一下
    combination = Combination(list_levels)
    for pair in combination:
        LSD(list_levels,list_total,pair[0],pair[1])

# 多重比较

more_result = pairwise_tukeyhsd(data['score'],data['method'])

'''
运行
'''

# 一、正态性检验

NormalTest(list_levels,0.05)
# 二、方差齐性检验
Levene_test(level1,level2,level3,alpha=0.05)

# 三、F检验
print(anova_result)

# 四、两两比较
Multiple_test(list_levels)
print(more_result)



