# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : t_two_pair

Description : 配对样本t检验

Author : leiliang

Date : 2020/7/8 4:54 下午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel
# import scipy
# from statsmodels.stats.diagnostic import lilliefors


# 二、相关性检验
def pearsonr_test(data):
    correlation, p_value = stats.pearsonr(data['x1'], data['x2'])
    return correlation, p_value


# 三、T检验
def t_two_pair_analysis(data):
    ttest, pval = ttest_rel(data['x1'], data['x2'])
    return ttest, pval


if __name__ == '__main__':
    data = pd.read_excel('pari_t.xlsx')
    x1 = data['x1']
    x2 = data['x2']
    x_all = [x1, x2]

    alpha = 0.05

    '''
    一、正态性检验
    '''
    # NormalTest(x_all, alpha=0.05)
    '''
    二、相关性检验
    '''
    correlation, p_value = stats.pearsonr(data['x1'], data['x2'])
    '''
    三、T检验
    '''
    ttest, pval = ttest_rel(data['x1'], data['x2'])
