# -*- coding: utf-8 -*-
import pandas as pd

import scipy.stats as stats
import logging

log = logging.getLogger(__name__)


def mannwhitneyu_test(x1, x2):
    alter = ['two-sided', 'greater', 'less']
    for i in alter:
        if i == 'two-sided':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x1, x2, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('双边检测：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)
        elif i == 'greater':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x1, x2, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('单侧检测，备择假设为">"：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)
        elif i == 'less':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x1, x2, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('单侧检测，备择假设为"<"：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)


def mannwhitneyu_test_with_diff(x):
    alter = ['two-sided', 'greater', 'less']
    res = []
    for i in alter:
        if i == 'two-sided':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('双边检测：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)
            res.append((wilcoxon_statistic, wilcoxon_pvalue))
        elif i == 'greater':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('单侧检测，备择假设为">"：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)
            res.append((wilcoxon_statistic, wilcoxon_pvalue))
        elif i == 'less':
            wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                 alternative=i)
            log.info('单侧检测，备择假设为"<"：')
            log.info(wilcoxon_statistic, wilcoxon_pvalue)
            res.append((wilcoxon_statistic, wilcoxon_pvalue))
    return [{"title": "Mann-Whitney U 检验 by 两样本差值"},
            {"row": ["双边检测", "单侧检测，备择假设为'>'", "单侧检测，备择假设为'<'"]},
            {"col": ["统计量", "P值"]},
            {"data": res}]


# 描述性统计分析
def nonparam_two_paired_describe_info(data, X, Y):
    data_groupby = data.groupby(X)
    new_data = pd.concat([data_groupby[Y[0]].count(), data_groupby[Y[0]].mean(),
                          data_groupby[Y[0]].std(),
                          data_groupby[Y[0]].std() / data_groupby[Y[0]].count()], axis=1)
    new_data.columns = ["count", "mean", "std", "std_err"]
    return {
        "row": new_data.index.values.tolist(),
        "col": new_data.columns.values.tolist(),
        "data": new_data.values.tolist(),
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/nonparametric_two_pair.csv')

    # 此代码中丢弃所有零差
    '''
    第一种情况：
    当用户输出两列数值型变量时
    '''
    x1 = data['x1']
    x2 = data['x2']
    log.info(mannwhitneyu_test(x1, x2))
    log.info("============================")
    '''
    第二种情况：
    当用户输出一列差值数值型变量时
    '''
    x = x1 - x2
    log.info(mannwhitneyu_test_with_diff(x))
