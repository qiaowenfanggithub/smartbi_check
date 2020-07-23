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
from utils import transform_h_table_data_to_v, format_dataframe


# 描述性统计分析
def t_two_paired_describe_info(data: pd.DataFrame, X, Y):
    data_groupby = data.groupby(X)
    new_data = pd.concat([data_groupby[Y[0]].count(), data_groupby[Y[0]].mean(),
                          data_groupby[Y[0]].std(),
                          data_groupby[Y[0]].std() / data_groupby[Y[0]].count()], axis=1)
    new_data.columns = ["个案数", "平均值", "标准偏差", "标准误差平均值"]
    new_data = format_dataframe(new_data, {"个案数": ".0f", "平均值": ".4f", "标准偏差": ".4f", "标准误差平均值": ".4f"})
    return {
        # 先支持一个配对的情况，因此行名前面先不加配对1
        "row": new_data.index.values.tolist(),
        "col": new_data.columns.values.tolist(),
        "data": new_data.values.tolist(),
        "title": "描述性统计分析"
    }


# 二、相关性检验
def pearsonr_test(*args, index=None, alpha=0.05):
    correlation, p_value = stats.pearsonr(*args)
    return {
        "row": [" & ".join(index)],
        "col": ["个案数", "相关性", "显著性", "拒绝原假设"],
        "data": ["{:.0f}".format(len(args[0])),
                 "{:.4f}".format(correlation),
                 "{:.4f}".format(p_value),
                 str(bool(p_value <= alpha))],
        "title": "配对样本相关性",
        "remark": "注：拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


# 三、T检验
def t_two_pair_analysis(*args, index=None, alpha=0.05):
    ttest, pval = ttest_rel(*args)
    data_cha = args[0] - args[1]
    cha_mean = data_cha.mean()
    cha_std = data_cha.std()
    cha_error_mean = data_cha.std() / np.sqrt(data_cha.count())  # 标准误差平均值
    cha_df = data_cha.count() - 1  # 自由度
    t = stats.t.ppf(alpha / 2, cha_df)
    cha_lower = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).min()
    cha_upper = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).max()
    return {
        "col": ["平均值", "标准偏差", "标准误差平均值",
                "差值{:.0%}置信区间下限".format(1-alpha),
                "差值{:.0%}置信区间上限".format(1-alpha),
                "t", "自由度", "P值(双尾)", "拒绝原假设"],
        "data": [["{:.0f}".format(cha_mean), "{:.0f}".format(cha_std),
                 "{:.0f}".format(cha_error_mean), "{:.0f}".format(cha_lower),
                 "{:.0f}".format(cha_upper), "{:.0f}".format(ttest),
                 "{:.0f}".format(cha_df), "{:.0f}".format(pval), str(bool(pval <= alpha))]],
        "row": [" & ".join(index)],
        "title": "配对样本检验",
        "remark": "注：拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/t_two_pair.csv')
    x1 = data['value1']
    x2 = data['value2']
    x_all = [x1, x2]
    #
    alpha = 0.06

    '''
    一、正态性检验
    '''
    # NormalTest(x_all, alpha=0.05)
    '''
    二、相关性检验
    '''
    print(pearsonr_test(*x_all, index=["value1", "value2"], alpha=alpha))
    '''
    三、T检验
    '''
    print(t_two_pair_analysis(*x_all, index=["value1", "value2"], alpha=alpha))

    # 统计描述
    # data, X, Y = transform_h_table_data_to_v(data, ["value1", "value2"])
    # print(t_two_paired_describe_info(data, X, Y))
