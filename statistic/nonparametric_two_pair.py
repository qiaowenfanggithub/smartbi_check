# -*- coding: utf-8 -*-
import pandas as pd
import scipy
import scipy.stats as stats
import os


# 描述性统计
def Wilcoxon_describe(data: pd.DataFrame,X):
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

# Wilcoxon 符号秩检验
def Wilcoxon_test(data: pd.DataFrame,X):
    alter = ['two-sided', 'greater', 'less']
    res = []
    row = []
    if len(X) == 1:
        x = data[X[0]]
        for i in alter:
            if i == 'two-sided':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                     alternative=i)
                row.append(['双边检测'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
            elif i == 'greater':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                     alternative=i)
                row.append(['单侧检测，备择假设为">"'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
            elif i == 'less':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(x, zero_method='wilcox', correction=False,
                                                                     alternative=i)
                row.append(['单侧检测，备择假设为"<"'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
    if len(X) == 2:
        v1 = data[X[0]]
        v2 = data[X[1]]
        for i in alter:
            if i == 'two-sided':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(v1, v2, zero_method='wilcox', correction=False,
                                                                     alternative=i)  # zero_method='wilcox' 丢弃所有零差
                row.append(['双边检测'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
            elif i == 'greater':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(v1, v2, zero_method='wilcox', correction=False,
                                                                     alternative=i)
                row.append(['单侧检测，备择假设为">"'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
            elif i == 'less':
                wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(v1, v2, zero_method='wilcox', correction=False,
                                                                     alternative=i)
                row.append(['单侧检测，备择假设为"<"'])
                res.append(["{:.4f}".format(wilcoxon_statistic), "{:.4f}".format(wilcoxon_pvalue)])
    return {
        'title': 'Wilcoxon 符号秩检验',
        'row': ['双侧检验','单侧检验，备择假设为“>”','单侧检验，备择假设为“<”'],
        'col': ['秩和','显著性'], # 这里返回的不是统计量，是正秩负秩中秩和较小的那个秩和
        'data': res
    }
if __name__ == '__main__':
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('twopair_feican.xlsx')
    v1 = data['x1']
    v2 = data['x2']
    X = ['x1', 'x2']
    r = Wilcoxon_test(data,X)
    print(r)

    d = Wilcoxon_describe(data,X)
    print(d)





# 此代码中丢弃所有零差
# 本地里源码里stats.wilcoxon()改了返回值，改成了返回z和p值，原来是返回较小的秩和，平台里的是没有改的，还是返回秩和
'''
第一种情况：
当用户输出两列数值型变量时
'''

'''
for i in alter:
    if i == 'two-sided':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x1,x2,zero_method='wilcox',correction=False,alternative=i) #zero_method='wilcox' 丢弃所有零差
        print('双边检测：')
        print(wilcoxon_statistic,wilcoxon_pvalue)
    elif i == 'greater':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x1,x2,zero_method='wilcox',correction=False,alternative=i)
        print('单侧检测，备择假设为">"：')
        print(wilcoxon_statistic,wilcoxon_pvalue)
    elif i == 'less':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x1,x2,zero_method='wilcox',correction=False,alternative=i)
        print('单侧检测，备择假设为"<"：')
        print(wilcoxon_statistic,wilcoxon_pvalue)

# 第二种情况：
# 当用户输出一列差值数值型变量时

x = x1-x2

for i in alter:
    if i == 'two-sided':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x,zero_method='wilcox',correction=False,alternative=i)
        print('双边检测：')
        print(wilcoxon_statistic,wilcoxon_pvalue)
    elif i == 'greater':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x,zero_method='wilcox',correction=False,alternative=i)
        print('单侧检测，备择假设为">"：')
        print(wilcoxon_statistic,wilcoxon_pvalue)
    elif i == 'less':
        wilcoxon_statistic,wilcoxon_pvalue = stats.wilcoxon(x,zero_method='wilcox',correction=False,alternative=i)
        print('单侧检测，备择假设为"<"：')
        print(wilcoxon_statistic,wilcoxon_pvalue)

'''



