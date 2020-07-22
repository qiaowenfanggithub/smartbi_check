# -*- coding: utf-8 -*-
from scipy.stats import ttest_1samp, stats
import numpy as np
import pandas as pd

'''
二、T检验
'''


def t_single_analysis(data, u, X, alpha=0.05):
    df = int(len(data)) - 1
    ttest, pval = ttest_1samp(data, u)  # t值和p值
    sample_mean = data.mean()  # 样本均值
    se = stats.sem(data)  # 样本标准误差平均值
    cha = sample_mean - u
    lower = (cha - se / np.sqrt(len(data) - 1))  # 差值的95%置信区间下限
    upper = (cha + se / np.sqrt(len(data) - 1))  # 差值的95%置信区间上限
    alpha_range = "{:.0f}".format((1 - alpha) * 100)
    return {
        "title": "单样本T检验",
        "row": X,
        "col": ["t值", "自由度", "p值", "拒绝原假设", "平均值差值",
                "差值的{}%置信区间上限".format(alpha_range), "差值的{}%置信区间下限".format(alpha_range)],
        "data": [["{:.4f}".format(ttest),
                  "{}".format(df),
                  "{:.4f}".format(pval),
                  bool(pval-alpha < 0),
                  "{:.4f}".format(sample_mean-u),
                  "{:.4f}".format(lower),
                  "{:.4f}".format(upper)]],
        "remark": "注：拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


"""
    描述性统计分析
"""


def t_single_describe_info(data, X):
    return {
        "title": "单样本统计",
        "row": X,
        "col": ["个案数", "平均值", "标准偏差", "标准误差平均值"],
        "data": [["{}".format(data[X[0]].count()), "{:.4f}".format(data[X[0]].mean()),
                 "{:.4f}".format(data[X[0]].std()), "{:.4f}".format(data[X[0]].std() / np.sqrt(data[X[0]].count()))]]
    }


if __name__ == '__main__':
    list = [2, 3, 4, 5, 6, 7]
    u = 4  # 总体均值
    '''
        一、正态性检验
        二、T检验
    '''
    alpha = 0.05
    data = pd.DataFrame({"X": list})
    print(t_single_analysis(data, u, ["X"], alpha=alpha))  # 这里只出了样本容量、T值、p值、样本均值。其他的统计量觉得不是特别必要,后面如果要求再加
