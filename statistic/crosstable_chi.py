# !/usr/bin/python3
# -*- coding: utf-8 -*-
from scipy.stats import chi2_contingency
import pandas as pd
import os
import xlrd
from statistic.utils import format_data,transform_table_data_to_html

def cross_chi2(index,columns):
    chi_res = []
    cross_result = pd.crosstab(index=index,columns=columns,margins=True)
    chi2_pearson, p_value_pearson, dof_pearson, expect_pearson = chi2_contingency(cross_result, correction=True,
                                                                              lambda_='pearson')  # pearson 卡方

    chi_res.append(["{:.4f}".format(chi2_pearson), "{:.4f}".format(p_value_pearson), dof_pearson])
    corss_index = cross_result.index.tolist()
    corss_index[-1] = '总计'
    corss_columns = cross_result.columns.tolist()
    corss_columns[-1] = '总计'
    corss_value = cross_result.values.tolist()

    expect = [format_data(pd.DataFrame(expect_pearson))]
    r1 = {
        'title':"交叉表",
        'row': corss_index,
        'col': corss_columns,
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
        'title': "Pearson卡方检验",
        'row': '',
        'col': ['值','显著性','自由度'],
        'data': chi_res
    }
    return [r1,r2,r3]




if __name__ == '__main__':
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('rc.xlsx')
    r = cross_chi2(data['c1'],data['c2'])
    print(r)