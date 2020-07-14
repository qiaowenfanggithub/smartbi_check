# -*- coding: utf-8 -*-
import pandas as pd
import scipy
import scipy.stats as stats
import logging

log = logging.getLogger(__name__)


def kruskal_test(*args):
    kw_statistic, kw_p = stats.kruskal(*args)
    log.info('use Kruskal-Wallis H 检验')
    return [{"title": "Kruskal-Wallis H 检验"},
            {"col": ["统计量", "P值"]},
            {"data": [kw_statistic, kw_p]}]


def median_test(*args, col_list=""):
    # 有nan返回nan
    ties = ['below', 'above', 'ignore']
    log.info('use median_test 检验')
    res = []
    col_name = col_list
    for i in ties:
        if i == 'below':
            stat, p, med, table = scipy.stats.median_test(*args, ties='below', nan_policy='propagate')
            log.info('med:{}'.format(med))
            log.info('列联表中，等于中位数的值不计算在内')
            log.info('table:{}'.format(table))
            res.append([{"info": "列联表中，等于中位数的值放下第二行"},
                        {"row": [">中位数", "<=中位数"]},
                        {"col": col_name},
                        {"data": table}])
            break
        elif i == 'above':
            stat, p, med, table = scipy.stats.median_test(*args, ties='above', nan_policy='propagate')
            log.info('med:{}'.format(med))
            log.info('列联表中，等于中位数的值不计算在内')
            log.info('table:{}'.format(table))
            res.append([{"info": "列联表中，等于中位数的值放下第二行"},
                        {"row": [">中位数", "<=中位数"]},
                        {"col": col_name},
                        {"data": table}])
        elif i == 'ignore':
            stat, p, med, table = scipy.stats.median_test(*args, ties='ignore', nan_policy='propagate')
            log.info('med:{}'.format(med))
            log.info('列联表中，等于中位数的值不计算在内')
            log.info('table:{}'.format(table))
            res.append([{"info": "列联表中，等于中位数的值放下第二行"},
                        {"row": [">中位数", "<=中位数"]},
                        {"col": col_name},
                        {"data": table}])
    return res


if __name__ == '__main__':
    logging.basicConfig(filename=None,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    data = pd.read_csv('./data/nonparametric_multi_independent.csv')

    # 数据中包含nan,返回nan

    x1 = data[data['level'] == 1]['value']
    x2 = data[data['level'] == 2]['value']
    x3 = data[data['level'] == 3]['value']

    # Kruskal-Wallis H 检验
    log.info("{}".format(kruskal_test(x1, x2, x3)))
    log.info("================================")

    # Median 中位数检验
    print(median_test(*[x1, x2, x3], col_list=["1", "2", "3"]))
