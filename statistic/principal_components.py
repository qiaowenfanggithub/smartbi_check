# !/usr/bin/python3
# -*- coding: utf-8 -*-
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bartlett
import math as math
from util import plot_and_output_base64_png, format_data_col, transform_table_data_to_html


# 相关系数矩阵
def correlation_matrix(x):
    x = x.astype(float)
    da = format_data_col(x.corr())
    col = da.columns.values.tolist()
    row = da.index.values.tolist()
    res = da.values.tolist()
    return transform_table_data_to_html({
        'title': "相关性矩阵",
        'col': col,
        'row':row,
        'data': res
    })

def kmo_Bartlett(x):
    x = x.astype(float)
    dataset_corr = x.corr()
    list = [dataset_corr.iloc[:,i] for i in range(dataset_corr.shape[1])]
    statistic,pvalue = bartlett(*list)
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr,ncol_inv_corr))
    for i in range(0,nrow_inv_corr,1):
        for j in range(i,ncol_inv_corr,1):
            A[i,j] = -(corr_inv[i,j])/(math.sqrt(corr_inv[i,i]*corr_inv[j,j]))
            A[j,i] = A[i,j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    # kmo_value = int(kmo_value)
    # statistic = int(statistic)
    # pvalue = int(pvalue)
    res = []
    res.append(["{:.4f}".format(kmo_value),"{:.4f}".format(statistic),"{:.4f}".format(pvalue)])

    col = ["KMO检验统计量","Bartlett's球状检验统计量","Bartlett's球状检验显著性"]
    title = "KMO检验和Bartlett's球状检验"
    return    {
        'title': title,
        'col': col,
        'data': res
    }


# PCA 过程
def PCA(x, components=None): # x 是接收的只包含特征变量的dataframe，components=None 接收的用户指定的主成分个数
    x = x.astype(float)
    result = []
    if components == None:
        components = int(x.size / len(x))  # 这里再考虑一下，接收用户指定的几个主成分
    ## 标准化
    average = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    r, c = np.shape(x)
    data_standardized = []
    mu = np.tile(average, (r, 1))  # r 行，铺一遍 https://www.cnblogs.com/elitphil/p/11824539.html
    data_standardized = (x - mu) / sigma
    ## 标准化

    cov_matrix = np.cov(data_standardized.T)  # 协方差矩阵
    EigenValue, EigenVector = np.linalg.eig(cov_matrix)  # 特征值和特征向量

    index = np.argsort(-EigenValue)  # 从大到小排序，返回的是元素在原有数据中的位置序号
    # Score = []
    selected_Vector = EigenVector.T[index[:components]]  # 根据指定的主成分个数，选择特征值相对应的特征向量
    Score = np.dot(data_standardized, selected_Vector.T)  # 计算主成分得分
    EigenValue_sorted = EigenValue[index] # 排序后的特征值

    '''
    特征值贡献及贡献率，需输出一个表
    '''
    EigenValue_contribution = pd.DataFrame(EigenValue_sorted, columns=['EigenValue'])
    EigenValue_contribution['Proportion'] = EigenValue_contribution['EigenValue'] / EigenValue_contribution['EigenValue'].sum()
    EigenValue_contribution['Cumulative'] = EigenValue_contribution['Proportion'].cumsum()

    '''
    碎石图和带有方差贡献率的碎石图，此图需输出
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6.8, 3)
    fig.subplots_adjust(wspace=0.4)
    ax1.plot(range(1, len(EigenValue_contribution) + 1), EigenValue_contribution['EigenValue'], 'o-')
    ax1.set_title('Scree Plot')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Eigenvalue')
    ax1.grid()

    ax2.plot(range(1, len(EigenValue_contribution) + 1), EigenValue_contribution['Proportion'], 'o-')
    ax2.plot(range(1, len(EigenValue_contribution) + 1), EigenValue_contribution['Cumulative'], 'bo-.')
    ax2.set_title('Variance Explained')
    ax2.set_xlabel('Principal Components')
    ax2.set_ylabel('Proportion')
    ax2.grid()
    plt.show()



    '''
    对应的特征向量
    '''
    vector_index = ['prin%d' % (i + 1) for i in range(len(selected_Vector))]
    vector_columns = x.columns.values.tolist()
    principal_vector = pd.DataFrame(selected_Vector, index=vector_index, columns=vector_columns).T

    '''
    主成分载荷(成分矩阵)，需输出一个表
    '''
    principal_component_load = pd.DataFrame()
    for i in range(len(selected_Vector)):
        principal_component_load['z%d' % (i + 1)] = np.sqrt(EigenValue_contribution['EigenValue'][i]) * principal_vector['prin%d' % (i + 1)]


    '''
    主成分得分(成分得分系数矩阵)
    '''
    principal_scores = pd.DataFrame()
    for i in range(len(selected_Vector)):
        principal_scores['prin%d_score' % (i + 1)] = Score[:, i]
    EigenValue_sorted_selected = EigenValue_sorted[:len(selected_Vector)]
    chengji = EigenValue_sorted_selected * principal_scores
    principal_scores['scores'] = chengji.sum(axis=1)
    principal_scores = principal_scores.sort_values(by='scores', ascending=False)

    Eig_contri = EigenValue_contribution
    Eig_contri['EigenValue'] = Eig_contri['EigenValue'].apply(lambda x: format(x, '.4f'))
    Eig_contri['Proportion'] = Eig_contri['Proportion'].apply(lambda x: format(x, '.2%'))
    Eig_contri['Cumulative'] = Eig_contri['Cumulative'].apply(lambda x: format(x, '.2%'))
    result.append({
        'title': "总方差解释",
        'col': ['特征值', '特征值方差贡献率', '累计方差贡献率'],
        'data': Eig_contri.values.tolist()
    })
    result.append({
        "title": "碎石图",
        "base64": "{}".format(plot_and_output_base64_png(plt))
    })
    prin_com_load = format_data_col(principal_component_load)
    col = prin_com_load.columns.values.tolist()
    row = prin_com_load.index.values.tolist()
    res = prin_com_load.values.tolist()
    result.append(transform_table_data_to_html({
        'title': "主成分载荷",
        'col': col,
        'row':row,
        'data': res
    }))

    prin_scores = format_data_col(principal_scores)
    col = prin_scores.columns.values.tolist()
    row = prin_scores.index.values.tolist()
    res = prin_scores.values.tolist()
    result.append(transform_table_data_to_html({
        'title': "主成分得分系数矩阵",
        'col': col,
        'row': row,
        'data': res
    }))

    return result

if __name__ == '__main__':

    import os
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('principal.xlsx')
    corrm = correlation_matrix(data)
    kmor = kmo_Bartlett(data)
    r = PCA(data, components=3)
    print(corrm)
    print(kmor)
    print(r)


