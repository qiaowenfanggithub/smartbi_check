# !/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors
import scipy.stats as stats
import scipy

import os

os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')

data = pd.read_excel('two_m.xlsx')

# 1、两个样本量数据可不一样
# 2、方差可不齐

'''
准备数据
'''
group1 = data[data['group'] == 1]['score']
group2 = data[data['group'] == 2]['score']
groups = [group1, group2]
'''
正态性检验
'''
alpha = 0.05


def check_normality(testData, alpha=0.05):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(testData) < 50:
        normaltest_statistic, normaltest_p = stats.normaltest(
            testData)  # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.normaltest.html
        print(normaltest_statistic, normaltest_p)
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
        shapiro_statistic, shapiro_p = stats.shapiro(
            testData)  # Perform the Shapiro-Wilk test for normality. https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.shapiro.html
        print(shapiro_statistic, shapiro_p)
        if shapiro_p < alpha:
            print("use shapiro:")
            print("data are not normal distributed")
            return False
        else:
            print("use shapiro:")
            print("data are normal distributed")
            return True
    if 300 >= len(testData) >= 50:
        lilliefors_statistic, lilliefors_p = lilliefors(
            testData)  # https://blog.csdn.net/qq_20207459/article/details/103000285
        print(lilliefors_statistic, lilliefors_p)
        if lilliefors_p < alpha:
            print("use lillifors:")
            print("data are not normal distributed")
            return False
        else:
            print("use lillifors:")
            print("data are normal distributed")
            return True
    if len(testData) > 300:
        kstest_statistic, kstest_p = scipy.stats.kstest(testData, 'norm')
        print(kstest_statistic, kstest_p)
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
def NormalTest(list_groups, alpha):
    for group in list_groups:
        # 正态性检验
        status = check_normality(group, alpha)
        if status == False:
            return False


'''

缺失值默认返回nan;
p值为双尾值


二、方差齐性检验
三、T检验
'''


def two_mean(group1, group2, a):
    # a = 0.05
    lev, levp = stats.levene(group1, group2)
    print(lev, levp)  # 输出方差齐性检验的统计量和P值，后续再优化
    if levp > a:
        print('方差相等')  # 后续优化
        tv, tp = stats.ttest_ind(group1, group2)
        print(tv, tp)
        if tp <= a:
            print('拒绝原假设，两个总体均值有显著差异')  # 后续优化
        else:
            print('不能拒绝原假设，两个总体均值无显著差异')  # 后续优化
    else:
        print('方差不等')  # 后续优化
        tv1, tp1 = stats.ttest_ind(group1, group2, equal_var=False)
        print(tv1, tp1)
        if tp1 <= a:
            print('拒绝原假设，两个总体均值有显著差异')  # 后续优化
        else:
            print('不能拒绝原假设，两个总体均值无显著差异')  # 后续优化


# 正态性检验
NormalTest(groups, alpha=0.05)
# T检验
two_mean(group1, group2, 0.05)
