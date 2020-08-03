# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from util import format_dataframe

log = logging.getLogger(__name__)


def t_two_independent_analysis(*args, alpha=0.05):
    data = []
    mean_diff = args[0].mean() - args[1].mean()
    std_diff = args[0].std() - args[1].std()
    data_cha = args[0].values - args[1].values
    cha_mean = data_cha.mean()
    cha_error_mean = data_cha.std() / np.sqrt(len(data_cha))  # 标准误差平均值
    cha_df = len(data_cha) - 1  # 自由度
    t = stats.t.ppf(alpha / 2, cha_df)
    cha_lower = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).min()
    cha_upper = pd.Series([cha_mean - t * cha_error_mean, cha_mean + t * cha_error_mean]).max()
    lev, levp = stats.levene(*args)
    # 输出方差齐性检验的统计量和P值，后续再优化
    log.info("方差齐性检验的统计量:{}, P值:{}".format(lev, levp))
    for b in [True, False]:
        log.info('方差相等')  # 后续优化
        tv, tp = stats.ttest_ind(*args, equal_var=b)
        if tp <= alpha:
            log.info('拒绝原假设，两个总体均值有显著差异')  # 后续优化
            data.append(
                ["{:.4f}".format(lev), "{:.4f}".format(levp),
                 str(b), "{:.4f}".format(tv), "{:.0f}".format(len(args[0]) - 1),
                 "{:.4f}".format(tp), "False", "{:.4f}".format(mean_diff),
                 "{:.4f}".format(std_diff), "{:.4f}".format(cha_lower),
                 "{:.4f}".format(cha_upper)])
        else:
            log.info('不能拒绝原假设，两个总体均值无显著差异')  # 后续优化
            data.append(
                ["{:.4f}".format(lev), "{:.4f}".format(levp),
                 str(b), "{:.4f}".format(tv), "{:.0f}".format(len(args[0]) - 1),
                 "{:.4f}".format(tp), "True", "{:.4f}".format(mean_diff),
                 "{:.4f}".format(std_diff), "{:.4f}".format(cha_lower),
                 "{:.4f}".format(cha_upper)])
    return {
        "row": ["假定等方差", "不假定等方差"],
        "col": ["F", "显著性", "拒绝原假设", "t", "自由度",
                "sig.(双尾)", "拒绝原假设", "平均值差值", "标准误差差值",
                "差值{:.0%}置信区间下限".format(1 - alpha), "差值{:.0%}置信区间下限".format(1 - alpha)],
        "data": data,
        "title": "独立样本T检验",
        "remarks": "注：拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


# 统计描述分析
def t_two_independent_describe_info(data: pd.DataFrame, X, Y):
    data_groupby = data.groupby(X)
    new_data = pd.concat([data_groupby[Y[0]].count(), data_groupby[Y[0]].mean(),
                          data_groupby[Y[0]].std(),
                          data_groupby[Y[0]].std() / data_groupby[Y[0]].count()], axis=1)
    new_data.columns = ["个案数", "平均值", "标准偏差", "标准误差平均值"]
    new_data = format_dataframe(new_data, {"个案数": ".0f", "平均值": ".4f", "标准偏差": ".4f", "标准误差平均值": ".4f"})
    return {
        "row": new_data.index.values.tolist(),
        "col": new_data.columns.values.tolist(),
        "data": new_data.values.tolist(),
        "title": "组统计"
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/t_two_independent.csv', header=0)

    # 1、两个样本量数据可不一样
    # 2、方差可不齐

    '''
    准备数据
    '''
    group1 = data[data['level'] == 1]['value']
    group2 = data[data['level'] == 2]['value']
    groups = [group1, group2]
    '''
    正态性检验
    '''
    alpha = 0.05

    # 正态性检验
    # NormalTest(groups, alpha=0.05)
    # T检验
    print(t_two_independent_analysis(*[group1, group2], alpha=alpha))

    # 统计描述分析
    # print(t_two_independent_describe_info(data, ["level"], ["value"]))
