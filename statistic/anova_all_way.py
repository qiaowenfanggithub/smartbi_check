# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : single_samples_t

Description : 

Author : leiliang

Date : 2020/7/1 3:30 下午

--------------------------------------------------------

"""
# !/usr/bin/python3
# -*- coding: utf-8 -*-
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import lilliefors
import scipy.stats as stats
import scipy
import pandas as pd
from itertools import combinations
import copy

import os

'''
一、正态性检验
'''


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
二、方差齐性检验   
'''


def Levene_test(*args, alpha=0.05):
    leveneTest_statistic, leveneTest_p = scipy.stats.levene(*args)
    print(leveneTest_statistic, leveneTest_p)
    if leveneTest_p < alpha:
        print("variances of groups are not equal")
        return False
    else:
        print("variances of groups are equal")
        return True


'''
三、F检验/ANOVA 表/ 主体效应检验
输出方差分析表 主体间效应检验
            df       sum_sq      mean_sq          F    PR(>F)
x1         1.0  3850.666667  3850.666667  42.864564  0.000002
x2         1.0  1089.000000  1089.000000  12.122449  0.002353
x1:x2      1.0  1225.000000  1225.000000  13.636364  0.001441
Residual  20.0  1796.666667    89.833333        NaN       NaN

x1 x2 x1:x2 对应的p值都小于0.05，所以都对y有显著性影响
'''


def anova_analysis_multivariate(data, X, Y):
    # 公式 因变量~ 自变量1+自变量2+ 自变量1和2的交互效应，这里是全模型
    x_formula = X + [":".join([c[0], c[1]]) for c in combinations(X, 2)]
    formula = '{}~{}'.format(Y[0], "+".join(x_formula))
    model = ols(formula, data).fit()
    anova_results = anova_lm(model)
    return anova_results.to_dict()


'''
四、多重比较
'''


def multiple_test_multivariate(data, X, Y, alpha = 0.05):
    """
    多因素方差分析-多重比较
    :param data: dataframe原始数据
    :param X: 自变量字段list
    :param Y: 因变量字段list
    :return: 所有因素组合的多重比较结果
    """

    def convert_bool_to_str(res_tmp):
        """pairwise_tukeyhsd返回里面的bool_不能序列化，转成str"""
        res_summary = res_tmp.summary().data
        for r in range(1, len(res_summary)):
            res_summary[r][-1] = str(res_summary[r][-1])
        return res_summary

    y_all = data[Y[0]]
    res_with_single_x = []
    for x in X:
        hsd_res = convert_bool_to_str(pairwise_tukeyhsd(y_all, data[x], alpha=alpha))
        res_with_single_x.append((x, hsd_res))

    try:
        from functools import reduce
    except ImportError as e:
        raise e

    multi_x_fn = lambda x, code=',': reduce(lambda x, y: [str(i) + code + str(j) for i in x for j in y], x)
    level_list = [data[x].unique() for x in X]
    level_combination = multi_x_fn(level_list, code=">>")
    res_with_multi_x = []
    for m in level_combination:
        data_x = copy.deepcopy(data)
        count = 0
        for x in m.split(">>")[:-1]:
            data_x = data_x[data_x[X[count]] == x]
            count += 1
        data_y = data_x[Y[0]]
        data_x = data_x[X[count]]
        hsd_multi = convert_bool_to_str(pairwise_tukeyhsd(data_y, data_x, alpha=alpha))
        res_with_multi_x.append((m, hsd_multi))
    return {"single_x_res": res_with_single_x, "multi_x_res": res_with_multi_x}


if __name__ == '__main__':
    '''
    准备数据
    '''
    data = pd.read_csv('./data/anova_all_way.csv')
    X = ["培训前成绩等级", "培训方法"]
    Y = ["成绩"]

    x1 = data['培训前成绩等级']
    x2 = data['培训方法']
    y = data['成绩'].values  # 类型是array
    x1_level1 = data[data['培训前成绩等级'] == 1]['成绩']
    x1_level2 = data[data['培训前成绩等级'] == 2]['成绩']
    x1_levels = [x1_level1, x1_level2]

    x2_level1 = data[data['培训方法'] == 1]['成绩']
    x2_level2 = data[data['培训方法'] == 2]['成绩']
    x2_level3 = data[data['培训方法'] == 3]['成绩']
    x2_levels = [x2_level1, x2_level2, x2_level3]
    y_total = data['成绩']

    '''
    一、正态性检验
    '''
    alpha = 0.05

    # 正态性检验
    # NormalTest(x1_levels, 0.05)
    # NormalTest(x2_levels, 0.05)

    # 方差齐性检验
    # Levene_test(x1_level1, x1_level2, alpha=0.05)
    # Levene_test(x2_level1, x2_level2, x2_level3, alpha=0.05)

    # F检验
    # anova_results = anova_analysis_multivariate(data, X, Y)

    # 多重比较
    multiple_test_multivariate(data, X, Y)
