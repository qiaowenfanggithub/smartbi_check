# -*- coding: utf-8 -*-
from scipy.stats import ttest_1samp, stats
import numpy as np

'''
二、T检验
'''


def t_single_analysis(data, u):
    sample_num = len(data)
    ttest, pval = ttest_1samp(data, u)  # t值和p值
    ttest = float("{:0.4f}".format(ttest))
    pval = float("{:0.4f}".format(pval))
    sample_mean = float("{:0.4f}".format(data.mean()))  # 样本均值
    sample_std = float("{:0.4f}".format(data.std()))  # 样本标准差
    se = float("{:0.4f}".format(stats.sem(data)))  # 样本标准误差平均值
    cha = float("{:0.4f}".format(sample_mean - u))
    lower = float("{:0.4f}".format(cha - se / np.sqrt(len(data) - 1)))  # 差值的95%置信区间下限
    upper = float("{:0.4f}".format(cha + se / np.sqrt(len(data) - 1)))  # 差值的95%置信区间上限
    return [("样本个数", sample_num), ("t值", ttest), ("p值", pval),
            ("样本均值", sample_mean), ("样本标准差", sample_std), ("样本标准误差平均值", se),
            ("差值的95%置信区间上限", upper), ("差值的95%置信区间下限", lower)]


"""
    描述性统计分析
"""


def t_single_describe_info(data, X):
    return {
        "row": X[0],
        "col": ["count", "mean", "std", "std_err"],
        "data": [data[X[0]].count(), data[X[0]].mean(),
                 data[X[0]].std(), data[X[0]].std() / np.sqrt(data[X[0]].count())]
    }


if __name__ == '__main__':
    list = [2, 3, 4, 5, 6, 7]
    u = 4  # 总体均值
    '''
        一、正态性检验
        二、T检验
    '''
    alpha = 0.05
    print(t_single_analysis(list, u))  # 这里只出了样本容量、T值、p值、样本均值。其他的统计量觉得不是特别必要,后面如果要求再加
