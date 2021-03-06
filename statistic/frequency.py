# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""

--------------------------------------------------------

File Name : frequency

Description :

Author : qiaowenfang

Date : 2020/8/19 3:26 下午

--------------------------------------------------------

"""
import pandas as pd
from util import format_dataframe


def data_frequency(data, X):
    col = ['频数','百分比%','累计百分比%']
    res = []
    for i in range(len(X)):
        result = data[X[i]].groupby(data[X[i]]).agg(['count'])
        result['百分比%'] = [i/sum(result['count']) for i in result['count']]
        result['累计百分比%'] = result['百分比%'].cumsum()
        result.columns = ['频数','百分比%','累计百分比%']
        # result = format_dataframe(result, {"频数": ".0f", "百分比%": ".2f%", "累计百分比%": ".2f%"})
        result['频数'] = result['频数'].map(lambda x: format(x,".0f" ))
        result['百分比%'] = result['百分比%'].map(lambda x: "%.2f%%" % (x * 100))
        result['累计百分比%'] = result['累计百分比%'].map(lambda x: "%.2f%%" % (x * 100))
        row = result.index.values.tolist()
        da = result.values.tolist()
        res.append({
            'title': "%s频数分布表"%(X[i]),
            'row': row,
            'col': col,
            'data': da
        })
    return res

if __name__ == '__main__':

    data = pd.DataFrame({'x1': ['a', 'b', 'a', 'a'], 'x2': [3, 3, 4, 4]})
    X = ['x1','x2']
    r = data_frequency(data,X)
    print(r)







