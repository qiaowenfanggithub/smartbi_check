# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""

--------------------------------------------------------

File Name : 

Description :

Author : qiaowenfang

Date : 2020/8/19 3:26 下午

--------------------------------------------------------

"""
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import os
import xlrd
from util import format_data, transform_table_data_to_html, format_data_col,sum_data

# "method":["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","neyman","cressie-read"]
def cross_chis(index,columns,fenceng):
    chi_res = []
    expect = []

    # 多层交叉表
    cross_result = pd.crosstab(index=index,columns=columns,margins=True)
    corss_index = cross_result.index.tolist()
    corss_index[-1] = '总计'
    corss_columns = cross_result.columns.tolist()
    corss_columns[-1] = '总计'
    corss_value = cross_result.values.tolist()

    # 交叉表分析
    cr_re = pd.crosstab(index=index,columns=columns,margins=False) # 给模型的不能有汇总列
    first_index = np.unique(index[0])
    for i in first_index:
        chis_pearson, p_value_pearson, dof_pearson, expect_pearson = chi2_contingency(cr_re.loc[i, :], correction=True, lambda_='pearson')
        chis_log, p_value_log, dof_log,expect_log = chi2_contingency(cr_re.loc[i, :], correction=True, lambda_='log-likelihood')
        chis_ftukey, p_value_ftukey, dof_ftukey, expect_ftukey = chi2_contingency(cr_re.loc[i, :], correction=True,lambda_='freeman-tukey')
        chis_mll, p_value_mll, dof_mll, expect_mll = chi2_contingency(cr_re.loc[i, :], correction=True,lambda_='mod-log-likelihood')
        chis_neyman, p_value_neyman, dof_neyman, expect_neyman = chi2_contingency(cr_re.loc[i, :], correction=True,lambda_='neyman')
        chis_cr, p_value_cr, dof_cr, expect_cr = chi2_contingency(cr_re.loc[i, :], correction=True,lambda_='cressie-read')

        chi_res.append(["{:.4f}".format(chis_pearson), "{:.4f}".format(p_value_pearson), dof_pearson])
        chi_res.append(["{:.4f}".format(chis_log), "{:.4f}".format(p_value_log), dof_log])
        chi_res.append(["{:.4f}".format(chis_ftukey), "{:.4f}".format(p_value_ftukey), dof_ftukey])
        chi_res.append(["{:.4f}".format(chis_mll), "{:.4f}".format(p_value_mll), dof_mll])
        chi_res.append(["{:.4f}".format(chis_neyman), "{:.4f}".format(p_value_neyman), dof_neyman])
        chi_res.append(["{:.4f}".format(chis_cr), "{:.4f}".format(p_value_cr), dof_cr])

        for j in expect_pearson:
            expect.append(j)
        # expect.extend(expect_pearson.tolist())
    expect = pd.DataFrame(expect)#.astype(float)
    expect = sum_data(expect)
    expect = format_data_col(expect).values.tolist()
    # row = ["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","neyman","cressie-read"]*len(first_index)
    row = []

    method = ["pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"]
    for uindex in first_index:
        for m in method:
            row.append(fenceng[0]+'_'+uindex + ':' + m)

    r1 = {
        'title':"交叉表",
        'row': ["/".join(["{}".format(d) for d in c]) for c in corss_index],
        'col': corss_columns,
        'data': corss_value
    }
    r1 = transform_table_data_to_html(r1)

    r2 = {
        'title': "期望频数表",
        'row': ["/".join(["{}".format(d) for d in c]) for c in corss_index],
        'col': corss_columns[1:],
        'data': expect
    }
    r2 = transform_table_data_to_html(r2)
    r3 = {
        'title': "卡方检验",
        'row':row,
        'col': ['值', '显著性', '自由度'],
        'data': chi_res
    }
    r3 = transform_table_data_to_html(r3)
    return  [r1, r2, r3] # expect


if __name__ == '__main__':
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('crosstable.xlsx')
    r = cross_chis([data['c1'],data['c4']],data['c2'],['c1'])
    print(r)