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
import numpy as np
import pandas as pd
from itertools import combinations
import copy
from anova_one_way import normal_test, levene_test
from util import format_dataframe

import os


# 多因素方差分析因子排列组合
def levels_conbination(data, X):
    try:
        from functools import reduce
    except ImportError as e:
        raise e
    combination_fn = lambda x: reduce(lambda x, y: [(i, j) for i in x for j in y], x)
    level_list = [data[x].unique() for x in X]
    level_combination_res = combination_fn(level_list)
    return level_combination_res


# 主体间因子
def level_info(data, X):
    res = []
    level_list = levels_conbination(data, X)
    data_group = data.groupby(X).indices
    for idx, level in enumerate(level_list):
        row = [str(l) for l in level] + [str(len(data_group[level]))]
        res.append(row)
    return {
        "col": ["因子", "因子水平" * (len(X) - 1), "个案数"],
        "data": res,
        "title": "主体间因子"
    }


# 正态性检验
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


# 对所有样本组进行正态性检验
# 先将各个样本分好，对所有样本检验正态性，也是对样本组里的每个样本检验
def normal_test_all(data, X, alpha=0.05):
    level_list = levels_conbination(data, X)
    data_group = data.groupby(X).indices
    level_data_list = [data_group[d] for d in level_list]
    normal_data = normal_test(level_list, level_data_list, alpha=alpha)
    for idx, l in enumerate(normal_data["data"]):
        normal_data["data"][idx] = [str(d) for d in level_list[idx]] + l
    return {
        "col": ["因子1", "因子2", "正态性检验统计", "显著性", "拒绝原假设"],
        "data": normal_data["data"],
        "title": "正态性检验",
        "remarks": "注：拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


'''
二、方差齐性检验   
'''


def levene_test_all(data, X, alpha=0.05):
    level_list = levels_conbination(data, X)
    data_group = data.groupby(X).indices
    level_data_list = [data_group[d] for d in level_list]
    return levene_test(*level_data_list, alpha=alpha)


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


def anova_analysis_multivariate(data, X, Y, alpha=0.05):
    # 公式 因变量~ 自变量1+自变量2+ 自变量1和2的交互效应，这里是全模型
    x_formula = X + [":".join([c[0], c[1]]) for c in combinations(X, 2)]
    formula = '{}~{}'.format(Y[0], "+".join(x_formula))
    model = ols(formula, data).fit()
    anova_results = anova_lm(model)
    anova_results["拒绝原假设"] = anova_results["PR(>F)"].map(lambda x: str(bool(x <= alpha)))
    anova_results = format_dataframe(anova_results, {"df": ".0f",
                                                     "sum_sq": ".4f",
                                                     "mean_sq": ".4f",
                                                     "F": ".4f",
                                                     "PR(>F)": ".4f"})
    anova_results[anova_results == "nan"] = ""
    return {
        "row": anova_results.index.tolist()[:-1] + ["总计"],
        "col": ["自由度", "平方和", "均方", "F", "显著性", "拒绝原假设"],
        "data": anova_results.values.tolist(),
        "title": "主体间效应检验"
    }


'''
四、多重比较
'''


# 描述性统计分析
def anova_all_way_describe_info(data: pd.DataFrame, X, Y):
    res = []
    level_list = levels_conbination(data, X)
    data_group = data.groupby(X).indices
    for idx, level in enumerate(level_list):
        row = [str(l) for l in level]
        res.append(row + ["{:.4f}".format(data_group[level].mean()),
                          "{:.4f}".format(data_group[level].std()),
                          "{:.0f}".format(len(data_group[level]))])
        if idx == (len(level_list) - 1):
            break
        if level_list[idx + 1][0] != level[0]:
            row_tmp = ["总计"] + [""] * (len(X) - 1)
            sum_data = data[data[X[0]] == level[0]][Y[0]]
            res.append(row_tmp + ["{:.4f}".format(sum_data.mean()),
                                  "{:.4f}".format(sum_data.std() / np.sqrt(sum_data.count())),
                                  "{:.0f}".format(sum_data.count())])
    return {
        "col": X + ["平均值", "标注偏差", "个案数"],
        "data": res,
        "title": "描述统计"
    }


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

    # 因子水平排列组合结果
    # print(levels_conbination(data, X))

    # 主体间因子
    # print(level_info(data, X))

    # 正态性检验
    # print(normal_test_all(data, X, 0.05))

    # 方差齐性检验
    print(levene_test_all(data, X, alpha=0.05))

    # F检验
    # print(anova_analysis_multivariate(data, X, Y))

    # 多重比较
    # multiple_test_multivariate(data, X, Y)

    # 描述性统计分析
    # print(anova_all_way_describe_info(data, ["培训前成绩等级", "培训方法"], ["成绩"]))
