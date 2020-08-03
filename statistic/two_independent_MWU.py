# -*- coding: utf-8 -*-

import pandas as pd
import scipy.stats as stats

'''
描述性统计
'''

def Mann_Whitney_U_describe(data: pd.DataFrame,X):
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


'''
Mann-Whitney U 检验
'''

def Mann_Whitney_U_test(data: pd.DataFrame,X):
    x1 = data[X[0]]
    x2 = data[X[1]]
    alter = ['two-sided', 'greater', 'less']
    res = []
    for i in alter:
        if i == 'two-sided':
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            res.append(["{:.4f}".format(mannwhitneyu_statistic), "{:.4f}".format(mannwhitneyu_pvalue)])
        elif i == 'greater': #单侧检测，备择假设为">"
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            res.append(["{:.4f}".format(mannwhitneyu_statistic), "{:.4f}".format(mannwhitneyu_pvalue)])
        elif i == 'less': #单侧检测，备择假设为"<"
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            res.append(["{:.4f}".format(mannwhitneyu_statistic), "{:.4f}".format(mannwhitneyu_pvalue)])
    return {
        'title': 'Mann-Whitney U 检验',
        'row': ['双侧检验','单侧检验，备择假设为“>”','单侧检验，备择假设为“<”'],
        'col': ['统计量','渐进显著性'],
        'data': res
    }



'''
wilcoxon 秩和检验
'''
# wilcoxon_ranksums_statistic,wilcoxon_ranksums_p = stats.ranksums(x1,x2) # wilcoxon 秩和检验 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html
# print('use Wilcoxon秩和检验')
# print('双边检测')
# print(wilcoxon_ranksums_statistic,wilcoxon_ranksums_p)




if __name__ == '__main__':
    import os
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')

    data = pd.read_excel('two_independ_feican.xlsx')
    x1 = data['x1']
    x2 = data['x2']
    X = ['x1', 'x2']

    r = Mann_Whitney_U_test(data,X)
    print(r)





