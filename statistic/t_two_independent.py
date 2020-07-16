# -*- coding: utf-8 -*-
import scipy.stats as stats
import pandas as pd
import logging

log = logging.getLogger(__name__)


def t_two_independent_analysis(group1, group2, alpha=0.05):
    lev, levp = stats.levene(group1, group2)
    log.info(lev, levp)  # 输出方差齐性检验的统计量和P值，后续再优化
    if levp > alpha:
        log.info('方差相等')  # 后续优化
        tv, tp = stats.ttest_ind(group1, group2)
        log.info(tv, tp)
        if tp <= alpha:
            log.info('拒绝原假设，两个总体均值有显著差异')  # 后续优化
            return True, lev, levp, tv, tp, False
        else:
            log.info('不能拒绝原假设，两个总体均值无显著差异')  # 后续优化
            return True, lev, levp, tv, tp, True
    else:
        log.info('方差不等')  # 后续优化
        tv1, tp1 = stats.ttest_ind(group1, group2, equal_var=False)
        log.info(tv1, tp1)
        if tp1 <= alpha:
            log.info('拒绝原假设，两个总体均值有显著差异')  # 后续优化
            return False, "", "", tv1, tp1, False
        else:
            log.info('不能拒绝原假设，两个总体均值无显著差异')  # 后续优化
            return False, "", "", tv1, tp1, True


# 统计描述分析
def t_two_independent_describe_info(data: pd.DataFrame, X, Y):
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
    # t_two_independent_analysis(group1, group2, 0.05)

    # 统计描述分析
    print(t_two_independent_describe_info(data, ["level"], ["value"]))
