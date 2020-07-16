# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : t_two_pair

Description : 配对样本t检验

Author : leiliang

Date : 2020/7/8 4:54 下午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel


# 描述性统计分析
def t_two_paired_describe_info(data: pd.DataFrame, X, Y):
    data_groupby = data.groupby(X)
    new_data = pd.concat([data_groupby[Y[0]].count(), data_groupby[Y[0]].mean(),
                          data_groupby[Y[0]].std(),
                          data_groupby[Y[0]].std() / data_groupby[Y[0]].count()], axis=1)
    new_data.columns = ["count", "mean", "std", "std_err"]
    return {
        "row": new_data.index.values.tolist(),
        "col": new_data.columns.values.tolist(),
        "data": new_data.values.tolist(),
        "title": "描述性统计分析"
    }


# 二、相关性检验
def pearsonr_test(*args, index=None):
    correlation, p_value = stats.pearsonr(*args)
    return {
        "row": " & ".join(index),
        "col": ["个案数", "相关性", "显著性"],
        "data": [len(args[0]), correlation, p_value],
        "title": "配对样本相关性"
    }


# 三、T检验
def t_two_pair_analysis(*args, index=None):
    ttest, pval = ttest_rel(*args)
    data['cha'] = args[0] - args[1]
    cha_mean = data['cha'].mean()
    cha_std = data['cha'].std()
    cha_error_mean = data['cha'].std() / np.sqrt(data['cha'].count())  # 标准误差平均值
    cha_df = data['cha'].count() - 1  # 自由度
    alpha = 0.05
    t = stats.t.ppf(alpha / 2, cha_df)
    cha_lower = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).min()
    cha_upper = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).max()
    return {
        "row": " - ".join(index),
        "col": ["平均值", "标准偏差", "配对差值-标准误差平均值", "差值95%置信区间-下限", "差值95%置信区间-上限", "t", "自由度", "P值(双尾)"],
        "data": [cha_mean, cha_std, cha_error_mean, cha_lower, cha_upper, ttest, cha_df, pval],
        "title": "配对样本检验"
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/t_two_pair.csv')
    x1 = data['value1']
    x2 = data['value2']
    x_all = [x1, x2]

    alpha = 0.05

    '''
    一、正态性检验
    '''
    # NormalTest(x_all, alpha=0.05)
    '''
    二、相关性检验
    '''
    print(pearsonr_test(*x_all, index=["value1", "value2"]))
    '''
    三、T检验
    '''
    print(t_two_pair_analysis(*x_all, index=["value1", "value2"]))
