# !/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel
import scipy
from statsmodels.stats.diagnostic import lilliefors
import os

os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace')
data = pd.read_excel('pari_t.xlsx')
x1 = data['x1']
x2 = data['x2']
x_all = [x1, x2]
'''
一、正态性检验
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
二、相关性检验
'''
correlation, p_value = stats.pearsonr(data['x1'], data['x2'])
'''
三、T检验
'''
ttest, pval = ttest_rel(data['x1'], data['x2'])

print(data['x1'].mean(), data['x2'].mean())  # 均值
NormalTest(x_all, alpha=0.05)  # 正态性检验
print(correlation, p_value)  # 相关性检验，是pearson检验
print(ttest, pval)  # 配对样本T检验
