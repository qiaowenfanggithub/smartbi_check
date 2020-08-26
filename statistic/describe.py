# !/usr/bin/python3
# -*- coding: utf-8 -*-
# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""

--------------------------------------------------------

File Name : describe

Description :

Author : qiaowenfang

Date : 2020/8/17 5:59 下午

--------------------------------------------------------

"""
import pandas as pd

# 描述性统计
def description(data: pd.DataFrame,X):
    data = data.astype(float)
    res = []
    for i in range(len(X)):
        res.append(["{:.0f}".format(data[X[i]].count()),"{:.4f}".format(data[X[i]].mean()),"{:.4f}".format(data[X[i]].std()),"{:.4f}".format(data[X[i]].var()),
                    "{:.4f}".format(data[X[i]].std()/data[X[i]].mean()),"{:.4f}".format(data[X[i]].min())
            ,"{:.4f}".format(data[X[i]].quantile(0.25)),"{:.4f}".format(data[X[i]].quantile(0.50)),"{:.4f}".format(data[X[i]].quantile(0.75)), "{:.4f}".format(data[X[i]].max()),
                    "{:.4f}".format(data[X[i]].skew()),"{:.4f}".format(data[X[i]].kurt())])
    col = ['样本量','平均值','标准差','方差','变异系数','最小值','25百分位数','中位数','75百分位数','最大值','偏度','峰度']
    return {
        'title':'描述性统计',
        'row':X,
        'col':col,
        'data':res
    }
if __name__ == '__main__':
    data = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [2, 3, 4, 5]})
    X = ['x1', 'x2']
    re = description(data,X)
    print(re)










