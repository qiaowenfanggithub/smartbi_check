# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
# import itertools
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
import scipy, math
from itertools import combinations
import os
import logging
from statsmodels.stats.multicomp import pairwise_tukeyhsd

log = logging.getLogger(__name__)

'''
一、正态性检验
'''


def check_normality(testData, alpha=0.05):
    # 20<样本数<50用normal test算法检验正态分布性
    if 20 < len(testData) < 50:
        # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.normaltest.html
        normaltest_statistic, normaltest_p = stats.normaltest(testData)
        log.info("统计量:{},P值:{}".format(normaltest_statistic, normaltest_p))
        if normaltest_p < alpha:
            log.info('use normaltest')
            log.info('data are not normal distributed')
            return normaltest_statistic, normaltest_p, "normaltest", False
        else:
            log.info('use normaltest')
            log.info('data are normal distributed')
            return normaltest_statistic, normaltest_p, "normaltest", True
    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(testData) < 50:
        # Perform the Shapiro-Wilk test for normality. https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.shapiro.html
        shapiro_statistic, shapiro_p = stats.shapiro(testData)
        log.info("统计量:{},P值:{}".format(shapiro_statistic, shapiro_p))
        if shapiro_p < alpha:
            log.info("use shapiro:")
            log.info("data are not normal distributed")
            return shapiro_statistic, shapiro_p, "shapiro", False
        else:
            log.info("use shapiro:")
            log.info("data are normal distributed")
            return shapiro_statistic, shapiro_p, "shapiro", True
    if 300 >= len(testData) >= 50:
        # https://blog.csdn.net/qq_20207459/article/details/103000285
        lilliefors_statistic, lilliefors_p = lilliefors(testData)
        log.info("统计量:{},P值:{}".format(lilliefors_statistic, lilliefors_p))
        if lilliefors_p < alpha:
            log.info("use lillifors:")
            log.info("data are not normal distributed")
            return lilliefors_statistic, lilliefors_p, "lillifors", False
        else:
            log.info("use lillifors:")
            log.info("data are normal distributed")
            return lilliefors_statistic, lilliefors_p, "lillifors", True
    if len(testData) > 300:
        kstest_statistic, kstest_p = scipy.stats.kstest(testData, 'norm')
        log.info("统计量:{},P值:{}".format(kstest_statistic, kstest_p))
        if kstest_p < alpha:
            log.info("use kstest:")
            log.info("data are not normal distributed")
            return kstest_statistic, kstest_p, "kstest", False
        else:
            log.info("use kstest:")
            log.info("data are normal distributed")
            return kstest_statistic, kstest_p, "kstest", True


#  对所有样本组进行正态性检验
# 先将各个样本分好，对所有样本检验正态性，也是对样本组里的每个样本检验
def normal_test(index_list, list_groups, alpha=0.05):
    res = []
    for group in list_groups:
        # 正态性检验
        res_one_level = check_normality(group, alpha)
        res.append(res_one_level)
    return [{"title": "正态性检验"},
            {"row": index_list},
            {"col": ["统计量", "P值", "正太检验方法", "reject"]},
            {"data": res}]


'''
二、方差齐性检验   
'''


def levene_test(*args, alpha=0.05):
    leveneTest_statistic, leveneTest_p = scipy.stats.levene(*args)
    log.info(leveneTest_statistic, leveneTest_p)
    if leveneTest_p < alpha:
        log.info("variances of groups are not equal")
        return leveneTest_statistic, leveneTest_p, False
    else:
        log.info("variances of groups are equal")
        return leveneTest_statistic, leveneTest_p, True


'''
三、F检验/ANOVA 表
'''


# 单因素方差分析
def anova_analysis(data, level, value):
    model = ols('{} ~ C({})'.format(value, level), data).fit()
    anova_result = anova_lm(model)
    return anova_result.to_dict()


# 多因素方差分析
# formula = 'y~x1+x2+x1:x2'  # 公式 因变量~ 自变量1+自变量2+ 自变量1和2的交互效应
# model = ols(formula, data).fit()
# anova_results = anova_lm(model)
# log.info(anova_results)

'''
四、多重比较
'''


# 组内误差，也叫随机误差
def SE(level):
    se = 0
    mean1 = np.mean(level)
    for i in level:
        error = i - mean1
        se += error ** 2
    return se


# 组内误差平方和，也叫随机误差，是各个水平的误差平方和相加
def SSE(list_levels):
    sse = 0
    for every_level in list_levels:
        se = SE(every_level)
        sse += se
    return sse


# 组内误差均方
def MSE(list_levels, list_total):
    sse = SSE(list_levels)
    mse = sse / (len(list_total) - 1 * len(list_levels)) * 1.0
    return mse


# 排列组合函数
def Combination(list_levels):
    combination = []
    for i in range(1, len(list_levels) + 1):
        iter = combinations(list_levels, i)
        combination.append(list(iter))
    # 需要排除第一个和最后一个
    return combination[1:-1][0]


# 两两比较
def LSD(list_levels, list_total, sample1, sample2):  # p值怎么算
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    distance = abs(mean1 - mean2)
    log.info('distance:::', distance)
    # t检验自由度
    df = len(list_total) - 1 * len(list_levels)
    mse = MSE(list_levels, list_total)
    log.info('MSE:::', mse)
    t_half = stats.t(df).isf(alpha / 2)  # 由自由度计算T值
    log.info('t_half:::', t_half)
    lsd = t_half * math.sqrt(mse * (1.0 / len(sample1) + 1.0 / len(sample2)))
    log.info('LSD:::', lsd)
    if distance < lsd:  # 是根据lsd值判断是否显著
        log.info('There is no significant difference between：', sample1, sample2)  # 优化
    else:
        log.info('There is significant difference between:', sample1, sample2)  # 优化


# 多重比较
def multiple_test(data, alpha=0.05):
    """
    :param data: 输入dataframe格式数据
    :return: 输出列表结果，第一行是列头，其他行是数据
    """
    log.info('---------------------multiple test------------------')
    # list_total = data.iloc[:, -1]  # 这里需要注意一下
    # combination = Combination(list_levels)
    # for pair in combination:
    #     LSD(list_levels, list_total, pair[0], pair[1])
    res = pairwise_tukeyhsd(data.iloc[:, -1], data.iloc[:, 1], alpha=alpha)
    res_summary = res.summary().data
    for r in range(1, len(res_summary)):
        res_summary[r][-1] = str(res_summary[r][-1])
    return res_summary


'''
运行
'''
if __name__ == '__main__':
    '''
    准备数据
    '''
    data = pd.read_excel('./data/one_v.xlsx', index=False)
    level1 = data[data['method'] == 1]['score']
    level2 = data[data['method'] == 2]['score']
    level3 = data[data['method'] == 3]['score']
    level_index = [d for d in data["method"].unique()]
    list_levels = [level1, level2, level3]
    list_total = data['score']

    alpha = 0.05

    # 一、正态性检验
    normal_test(level_index, list_levels, alpha=0.05)

    # 二、方差齐性检验
    levene_test(level1, level2, level3, alpha=0.05)

    # 三、F检验
    anova_analysis(data, "score", "method")

    # 四、两两比较
    # Multiple_test(list_levels)
    res = multiple_test(data)
    print(res)