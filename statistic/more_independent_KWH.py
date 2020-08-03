# -*- coding: utf-8 -*-
import pandas as pd
import scipy
import scipy.stats as stats

import os


def Kruskal_Wallis_H_describe(data: pd.DataFrame,X):
    data = data.astype(float)
    res = []
    for i in range(len(X)):
        res.append(["{:.0f}".format(data[X[i]].count()),"{:.4f}".format(data[X[i]].mean()),"{:.4f}".format(data[X[i]].std()),"{:.4f}".format(data[X[i]].min()),
                "{:.4f}".format(data[X[i]].max()),"{:.4f}".format(data[X[i]].quantile(0.25)),"{:.4f}".format(data[X[i]].quantile(0.50)),"{:.4f}".format(data[X[i]].quantile(0.75))])
    col = ['个案数','平均值','标准偏差','最小值','最大值','25百分位数','50百分位数','75百分位数']
    return {
        'title':'描述性统计',
        'row':X,
        'col':col,
        'data':res
    }


def Kruskal_Wallis_H_test(data:pd.DataFrame,X):
    arg = []
    res = []
    for i in range(len(X)):
        arg.append(data[X[i]])
    kw_statistic, kw_p = stats.kruskal(*arg)
    res.append(["{:.4f}".format(kw_statistic), "{:.4f}".format(kw_p)])
    return {
        'title':'Kruskal-Wallis H 检验',
        'row':'',
        'col':['统计量','显著性'],
        'data':res
    }


'''
中位数检验

print('use median_test 检验')
ties  = ['below','above','ignore']
for i in ties:
    if i == 'below':
        stat,p,med,table = scipy.stats.median_test(x1,x2,x3,ties = 'below',nan_policy = 'propagate')
        print('median_test_statistic:', stat, 'median_test_p:', p)
        print('med:', med)
        print('列联表中，等于中位数的值放下第二行')
        print('table:', table)
    elif i == 'above':
        stat,p,med,table = scipy.stats.median_test(x1,x2,x3,ties = 'above',nan_policy = 'propagate')
        print('median_test_statistic:', stat, 'median_test_p:', p)
        print('med:', med)
        print('列联表中，等于中位数的值放下第一行')
        print('table:', table)
    elif i == 'ignore':
        stat, p, med, table = scipy.stats.median_test(x1, x2, x3, ties='above', nan_policy='propagate')
        print('median_test_statistic:', stat, 'median_test_p:', p)
        print('med:', med)
        print('列联表中，等于中位数的值不计算在内')
        print('table:', table)

'''



if __name__ == '__main__':


    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('more_independ_feican.xlsx')

    # 数据中包含nan,返回nan

    x1 = data[data['fam'] == 1]['creative'].values
    x2 = data[data['fam'] == 2]['creative'].values
    x3 = data[data['fam'] == 3]['creative'].values





