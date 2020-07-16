# !/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import t
import os
os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
data = pd.read_excel('pari_t.xlsx')

'''
配对样本T检验的结果统计量
'''

'''
一、第一张表
'''

# 平均值、个案数、标准偏差、标准误差平均值
x1_mean = data['x1'].mean()
x1_count = data['x1'].count()
x1_std = data['x1'].std()
x1_error_mean = data['x1'].std()/np.sqrt(data['x1'].count()) #标准误差平均值

x2_mean = data['x2'].mean()
x2_count = data['x2'].count()
x2_std = data['x2'].std()
x2_error_mean = data['x2'].std()/np.sqrt(data['x2'].count()) #标准误差平均值

print(x1_mean,x1_count,x1_std,x1_error_mean)
print(x2_mean,x2_count,x2_std,x2_error_mean)

'''
第二张表
'''
# 个案数 相关性 显著性
x1_x2_num = data.x1.count()
r,p = stats.pearsonr(data['x1'],data['x2'])
print(x1_x2_num,r,p)

'''
第三张表
'''
# 均值 标准偏差 配对差值标准误差平均值 差值95%置信区间下、上限 t值 自由度 sig(双尾)
data['cha'] = data['x1']-data['x2']
cha_mean = data['cha'].mean()
cha_std = data['cha'].std()
cha_error_mean = data['cha'].std()/np.sqrt(data['cha'].count()) #标准误差平均值
cha_df = data['cha'].count() - 1 # 自由度
alpha = 0.05
t = t.ppf(alpha/2,cha_df)
cha_lower = pd.Series([cha_mean - t*cha_error_mean,cha_mean + t*cha_error_mean]).min()
cha_upper = pd.Series([cha_mean - t*cha_error_mean,cha_mean + t*cha_error_mean]).max()

print(cha_mean,cha_std,cha_error_mean,cha_lower,cha_upper)

