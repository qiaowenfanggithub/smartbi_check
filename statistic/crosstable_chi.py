# !/usr/bin/python3
# -*- coding: utf-8 -*-
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import os
from util import format_data,transform_table_data_to_html,sum_data,format_data_col
from scipy.stats.contingency import expected_freq
def cross_chi2(index,columns):
    chi_res = []
    cross_result = pd.crosstab(index=index,columns=columns,margins=True)
    cr_re = pd.crosstab(index=index,columns=columns,margins=False) # 给模型的不能有汇总列，8/25修改
    chi2_pearson, p_value_pearson, dof_pearson, expect_pearson = chi2_contingency(cr_re, correction=True,lambda_='pearson')  # pearson 卡方
    chi2_log, p_value_log, dof_log, expect_log = chi2_contingency(cr_re, correction=True,lambda_='log-likelihood')
    chi2_ftukey, p_value_ftukey, dof_ftukey, expect_ftukey = chi2_contingency(cr_re, correction=True,lambda_='freeman-tukey')
    chi2_mll, p_value_mll, dof_mll, expect_mll = chi2_contingency(cr_re, correction=True,lambda_='mod-log-likelihood')
    chi2_neyman, p_value_neyman, dof_neyman, expect_neyman = chi2_contingency(cr_re, correction=True,lambda_='neyman')
    chi2_cr, p_value_cr, dof_cr, expect_cr = chi2_contingency(cr_re, correction=True, lambda_='cressie-read')

    chi_res.append(["{:.4f}".format(chi2_pearson), "{:.4f}".format(p_value_pearson), dof_pearson])
    chi_res.append(["{:.4f}".format(chi2_log), "{:.4f}".format(p_value_log), dof_log])
    chi_res.append(["{:.4f}".format(chi2_ftukey), "{:.4f}".format(p_value_ftukey), dof_ftukey])
    chi_res.append(["{:.4f}".format(chi2_mll), "{:.4f}".format(p_value_mll), dof_mll])
    chi_res.append(["{:.4f}".format(chi2_neyman), "{:.4f}".format(p_value_neyman), dof_neyman])
    chi_res.append(["{:.4f}".format(chi2_cr), "{:.4f}".format(p_value_cr), dof_cr])



    corss_index = cross_result.index.tolist()
    corss_index[-1] = '总计'
    corss_columns = cross_result.columns.tolist()
    corss_columns[-1] = '总计'

    corss_value = cross_result.values.tolist()
    exp = pd.DataFrame(expected_freq(cr_re))
    exp = sum_data(exp)
    expect = format_data_col(exp).values.tolist()


    r1 = {
        'title':"交叉表",
        'row': corss_index,
        'col': corss_columns[0:],
        'data': corss_value
    }
    r1 = transform_table_data_to_html(r1)

    r2 = {
        'title': "期望频数表",
        'row': corss_index,
        'col': corss_columns,
        'data': expect
    }
    r2 = transform_table_data_to_html(r2)
    r3 = {
        'title': "卡方检验",
        'row': ["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","neyman","cressie-read"],
        'col': ['值','显著性','自由度'],
        'data': chi_res
    }
    r3 = transform_table_data_to_html(r3)
    return [r1,r2,r3]




if __name__ == '__main__':
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('crosstable.xlsx')
    r = cross_chi2(data['c1'],data['c2'])
    print(r)