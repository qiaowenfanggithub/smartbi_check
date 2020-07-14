from scipy.stats import ttest_ind, norm, f
import numpy as np


def ftest(s1, s2):
    '''F检验样本总体方差是否相等'''
    print("Null Hypothesis:var(s1)=var(s2)，α=0.05")
    F = np.var(s1) / np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2 * abs(0.5 - f.cdf(F, v1, v2))
    print(p_val)
    if p_val < 0.05:
        print("Reject the Null Hypothesis.")
        equal_var = False
    else:
        print("Accept the Null Hypothesis.")
        equal_var = True
    return equal_var


def ttest_ind_fun(s1, s2):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(s1, s2)
    print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
    ttest, pval = ttest_ind(s1, s2, equal_var=equal_var)
    if pval < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval


np.random.seed(42)
s1 = norm.rvs(loc=1, scale=1.0, size=20)
s2 = norm.rvs(loc=1.5, scale=0.5, size=20)
s3 = norm.rvs(loc=1.5, scale=0.5, size=25)

ttest_ind_fun(s1, s2)
ttest_ind_fun(s2, s3)
