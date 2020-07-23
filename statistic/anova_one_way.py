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
from utils import transform_table_data_to_html

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
        if normaltest_p <= alpha:
            log.info('use normaltest')
            log.info('data are not normal distributed')
            return ["{:.4f}".format(normaltest_statistic), "{:.4f}".format(normaltest_p), "True"]
        else:
            log.info('use normaltest')
            log.info('data are normal distributed')
            return ["{:.4f}".format(normaltest_statistic), "{:.4f}".format(normaltest_p), "False"]
    # 样本数小于50用Shapiro-Wilk算法检验正态分布性
    if len(testData) < 50:
        # Perform the Shapiro-Wilk test for normality. https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.shapiro.html
        shapiro_statistic, shapiro_p = stats.shapiro(testData)
        log.info("统计量:{},P值:{}".format(shapiro_statistic, shapiro_p))
        if shapiro_p <= alpha:
            log.info("use shapiro:")
            log.info("data are not normal distributed")
            return ["{:.4f}".format(shapiro_statistic), "{:.4f}".format(shapiro_p), "True"]
        else:
            log.info("use shapiro:")
            log.info("data are normal distributed")
            return ["{:.4f}".format(shapiro_statistic), "{:.4f}".format(shapiro_p), "False"]
    if 300 >= len(testData) >= 50:
        # https://blog.csdn.net/qq_20207459/article/details/103000285
        lilliefors_statistic, lilliefors_p = lilliefors(testData)
        log.info("统计量:{},P值:{}".format(lilliefors_statistic, lilliefors_p))
        if lilliefors_p <= alpha:
            log.info("use lillifors:")
            log.info("data are not normal distributed")
            return ["{:.4f}".format(lilliefors_statistic), "{:.4f}".format(lilliefors_p), "True"]
        else:
            log.info("use lillifors:")
            log.info("data are normal distributed")
            return ["{:.4f}".format(lilliefors_statistic), "{:.4f}".format(lilliefors_p), "False"]
    if len(testData) > 300:
        kstest_statistic, kstest_p = scipy.stats.kstest(testData, 'norm')
        log.info("统计量:{},P值:{}".format(kstest_statistic, kstest_p))
        if kstest_p <= alpha:
            log.info("use kstest:")
            log.info("data are not normal distributed")
            return ["{:.4f}".format(kstest_statistic), "{:.4f}".format(kstest_p), "True"]
        else:
            log.info("use kstest:")
            log.info("data are normal distributed")
            return ["{:.4f}".format(kstest_statistic), "{:.4f}".format(kstest_p), "False"]


#  对所有样本组进行正态性检验
# 先将各个样本分好，对所有样本检验正态性，也是对样本组里的每个样本检验
def normal_test(index_list, list_groups, alpha=0.05):
    res = []
    for group in list_groups:
        # 正态性检验
        res_one_level = check_normality(group, alpha)
        res.append(res_one_level)
    return {"title": "正态性检验",
            "remarks": "注: 拒绝原假设, False表示不拒绝原假设, True表示拒绝原假设",
            "row": index_list,
            "col": ["正态性检验统计", "显著性", "拒绝原假设"],
            "data": res}


'''
二、方差齐性检验   
'''


def levene_test(*args, alpha=0.05):
    res = []
    row = []
    center_dict = {"mean": "基于平均值", "median": "基于中位数", "trimmed": "基于剪除后平均值"}
    for c in ["mean", "median", "trimmed"]:
        row.append(center_dict[c])
        leveneTest_statistic, leveneTest_p = scipy.stats.levene(*args, center=c)
        if leveneTest_p <= alpha:
            log.info("variances of groups are not equal")
            res.append(["{:.4f}".format(leveneTest_statistic), "{:.4f}".format(leveneTest_p), "True"])
        else:
            log.info("variances of groups are equal")
            res.append(["{:.4f}".format(leveneTest_statistic), "{:.4f}".format(leveneTest_p), "False"])
    return {
        "row": row,
        "col": ["Levene统计", "显著性", "拒绝原假设"],
        "data": res,
        "title": "方差齐性检验",
        "remarks": "注: 拒绝原假设, False表示不拒绝原假设, True表示拒绝原假设"
    }


'''
三、F检验/ANOVA 表
'''


# 单因素方差分析
def anova_analysis(data, level, value, alpha=0.05):
    model = ols('{} ~ C({})'.format(value, level), data).fit()
    anova_result = anova_lm(model)
    # anova_result.fillna("", inplace=True)
    anova_result.index = ["组间", "组内"]
    anova_result.columns = ["自由度", "平方和", "均方", "F", "显著性"]
    anova_result = anova_result.append(
        pd.DataFrame([[anova_result["自由度"].sum(), anova_result["平方和"].sum(), None, None, None]],
                     index=["总计"], columns=anova_result.columns))
    anova_result["自由度"] = anova_result["自由度"].astype("object")
    anova_result["拒绝原假设"] = pd.Series([str(bool(anova_result["显著性"][0] - alpha)), "", ""], index=["组间", "组内", "总计"])
    anova_result = anova_result.round({"平方和": 4, "均方": 4, "F": 4, "显著性": 4})
    anova_result.fillna("", inplace=True)
    return {
        "row": anova_result.index.tolist(),
        "col": anova_result.columns.tolist(),
        "data": anova_result.values.tolist(),
        "title": "ANOVA",
        "remarks": "注: 拒绝原假设, False表示不拒绝原假设, True表示拒绝原假设"
    }


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
def multiple_test(data, X, Y, alpha=0.05):
    """
    :param data: 输入dataframe格式数据
    :return: 输出列表结果，第一行是列头，其他行是数据
    """
    log.info('---------------------multiple test------------------')
    # list_total = data.iloc[:, -1]  # 这里需要注意一下
    # combination = Combination(list_levels)
    # for pair in combination:
    #     LSD(list_levels, list_total, pair[0], pair[1])
    alpha_range = (1 - alpha) * 100
    res = pairwise_tukeyhsd(data[Y[0]], data[X[0]], alpha=alpha)
    res_summary = res.summary().data
    for r in range(1, len(res_summary)):
        res_summary[r][-1] = str(res_summary[r][-1])
    col_map = {
        "group1": "组1",
        "group2": "组2",
        "meandiff": "平均值差值2-1",
        "p-adj": "显著性",
        "lower": "{}%置信区间下限".format(alpha_range),
        "upper": "{}%置信区间上限".format(alpha_range),
        "reject": "拒绝原假设",
    }
    return {
        "col": [col_map[c] for c in res_summary[0]],
        "data": res_summary[1:],
        "title": "多重比较",
        "remarks": "注：多重比较方法基于Tukey HSD。拒绝原假设，False表示不拒绝原假设，True表示拒绝原假设。"
    }


# 描述性统计分析数据
def anova_one_way_describe_info(data: pd.DataFrame, X, Y, alpha=0.05):
    # 个案数
    data_count_by_level = data.groupby(X)[Y[0]].count().astype("int16")
    data_count_total = data[Y[0]].count().astype("int16")
    # 均值
    data_mean_by_level = data.groupby(X)[Y[0]].mean()
    data_mean_total = data[Y[0]].mean()
    # 标准偏差
    data_std_by_level = data.groupby(X)[Y[0]].std()
    data_std_total = data[Y[0]].std()
    # 标准错误 = 标准偏差 / sqrt(个案数)
    data_std_err_by_level = data_std_by_level / np.sqrt(data_count_by_level)
    data_std_err_total = data_std_total / np.sqrt(data_count_total)
    # 置信区间下限
    data_df_by_level = data_count_by_level - 1
    data_df_by_level = data_df_by_level.apply(lambda x: stats.t.ppf(alpha / 2, x))
    data_df_by_level_sub = data_mean_by_level - data_df_by_level * data_std_err_by_level
    data_df_by_level_plus = data_mean_by_level + data_df_by_level * data_std_err_by_level
    data_df_by_level = pd.concat([data_df_by_level_sub, data_df_by_level_plus], axis=1)
    data_lower_by_level = data_df_by_level.apply(lambda x: min(x), axis=1)
    data_t = stats.t.ppf(alpha / 2, data_count_total - 1)
    data_lower_total = min(data_mean_total - data_t * data_std_err_total, data_mean_total + data_t * data_std_err_total)
    # 置信区间上限
    data_upper_by_level = data_df_by_level.apply(lambda x: max(x), axis=1)
    data_upper_total = max(data_mean_total - data_t * data_std_err_total, data_mean_total + data_t * data_std_err_total)
    # 最小值
    data_min_by_level = data.groupby(X)[Y[0]].min().astype("int16")
    data_min_total = data[Y[0]].min().astype("int16")
    # 最大值
    data_max_by_level = data.groupby(X)[Y[0]].max().astype("int16")
    data_max_total = data[Y[0]].max().astype("int16")
    new_data_by_level = pd.concat([data_count_by_level, data_mean_by_level,
                                   data_std_by_level, data_std_err_by_level,
                                   data_lower_by_level, data_upper_by_level,
                                   data_min_by_level, data_max_by_level], axis=1)
    alpha_range = str(round((1 - alpha) * 100, 2))
    # new_data_by_level.columns = ["个案数", "平均值", "标准偏差", "标准错误", "下限", "上限", "最小值", "最大值"]
    new_data_by_level.columns = ["个案数", "平均值", "标准偏差", "标准错误",
                                 "下限",
                                 "上限",
                                 "最小值", "最大值"]
    new_data_total = pd.DataFrame([[data_count_total, data_mean_total,
                                    data_std_total, data_std_err_total,
                                    data_lower_total, data_upper_total,
                                    data_min_total, data_max_total]],
                                  columns=["个案数", "平均值", "标准偏差", "标准错误",
                                           "下限".format(alpha_range),
                                           "上限".format(alpha_range),
                                           "最小值", "最大值"], index=["总计"])
    new_data = pd.concat([new_data_by_level, new_data_total], axis=0)
    new_data = new_data.round({"平均值": 4, "标准偏差": 4, "标准错误": 4, "下限": 4, "上限": 4})
    col_map = {
        "个案数": "个案数",
        "平均值": "平均值",
        "标准偏差": "标准偏差",
        "标准错误": "标准错误",
        "下限": "均值的{:}%置信区间-下限".format(alpha_range),
        "上限": "均值的{:}%置信区间-上限".format(alpha_range),
        "最小值": "最小值",
        "最大值": "最大值"
    }
    new_data[["个案数", "最大值", "最小值"]] = new_data[["个案数", "最大值", "最小值"]].astype("object")
    return {
        "row": new_data.index.values[:-1].tolist(),
        "col": [col_map[c] for c in new_data.columns.values.tolist()],
        "data": new_data.values.tolist(),
        "title": "描述性统计分析"
    }


'''
运行
'''
if __name__ == '__main__':
    '''
    准备数据
    '''
    data = pd.read_excel('./data/one_v.xlsx', index=False)
    data = data.astype(int)
    level1 = data[data['method'] == 1]['score']
    level2 = data[data['method'] == 2]['score']
    level3 = data[data['method'] == 3]['score']
    level_index = [d for d in data["method"].unique()]
    list_levels = [level1, level2, level3]
    list_total = data['score']

    alpha = 0.05

    # # 一、正态性检验
    # print(normal_test(level_index, list_levels, alpha=0.05))
    #
    # # 二、方差齐性检验
    # print(levene_test(level1, level2, level3, alpha=0.05))
    #
    # # 三、F检验
    # print(anova_analysis(data, "method", "score"))
    #
    # # 四、两两比较
    # # Multiple_test(list_levels)
    print(multiple_test(data, ["method"], ["score"], alpha=0.06))

    # 描述性统计分析
    # print(anova_one_way_describe_info(data, ["method"], ["score"], alpha=0.06))
    # print(res)
