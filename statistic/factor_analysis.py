# !/usr/bin/python3
# -*- coding: utf-8 -*-
# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""

--------------------------------------------------------

File Name : factor_analysis

Description :

Author : qiaowenfang

Date : 2020/8/19 5:19 下午

--------------------------------------------------------

"""
# 基于方差最大化正交旋转的因子分析
import numpy as np
import pandas as pd
from scipy.stats import bartlett
import math as math
import matplotlib.pyplot as plt
from util import format_data_col, transform_table_data_to_html, plot_and_output_base64_png


class FA(object):
    '''
    该类用于因子分析，有5个方法分别用于计算
    特征值及方差贡献率
    旋转前的因子载荷
    旋转后的因子载荷
    因子得分系数
    因子得分
    '''

    def __init__(self,component,gamma = 1.0, q = 20,tol = 1e-8, standardize = True, transpose = False):
        '''
        gamma = 1.0, q = 20,tol = 1e-8 均为最大方差正交旋转的参数，无特殊需求默认即可
        '''
        self.component = component
        self.gamma = gamma
        self.q = q
        self.tol = tol
        self.standardize = standardize
        self.transpose = transpose

    # 相关系数矩阵
    def correlation_matrix(self,data):
        data = data.astype(float)
        da = format_data_col(data.corr())
        col = da.columns.values.tolist()
        row = da.index.values.tolist()
        res = da.values.tolist()
        return transform_table_data_to_html({
            'title': "相关性矩阵",
            'col': col,
            'row': row,
            'data': res
        })

    # KMO检验和Bartlett's球状检验
    def kmo_Bartlett(self,data):
        data = data.astype(float)
        dataset_corr = data.corr()
        list = [dataset_corr.iloc[:, i] for i in range(dataset_corr.shape[1])]
        statistic, pvalue = bartlett(*list)
        corr_inv = np.linalg.inv(dataset_corr)
        nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        dataset_corr = np.asarray(dataset_corr)
        kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        kmo_value = kmo_num / kmo_denom
        res = []
        res.append(["{:.4f}".format(kmo_value), "{:.4f}".format(statistic), "{:.4f}".format(pvalue)])
        col = ["KMO检验统计量", "Bartlett's球状检验统计量", "Bartlett's球状检验显著性"]
        title = "KMO检验和Bartlett's球状检验"
        return {
            'title': title,
            'col': col,
            'data': res
        }

    # z标准化
    def standardization(self,data):
        z = (data-data.mean())/data.std(ddof=1)
        return z

    def var_contribution(self,data):
        data = data.astype(float)
        '''
        该方法用于输出特征值、方差贡献率以及累计方差贡献率
        '''
        # 将数据转存为数组形式方便操作
        var_name = data.columns
        data = np.array(data)
        if self.standardize == True:
            z = FA.standardization(self,data)
            # 按列求解相关系数矩阵，存入cor中（rowvar = 0指定按列求解相关系数矩阵）
            cor = np.corrcoef(z,rowvar=0)
            # 求解相关系数矩阵的特征值与特征向量，并按照特征值由大到小排序
            # 注意numpy 中求出的特征向量是按列排序而非行，因此注意将矩阵装置
            eigvalue,eigvector = np.linalg.eig(cor)
            eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue,dtype=float,columns=['eigvalue']))
            # 将特征值向量按特征值由大到小排序
            eigdf = eigdf.sort_values('eigvalue')[::-1]
            # 将调整好的特征向量存储在eigvalue
            eigvector = np.array(eigdf.iloc[:,:-1])
            # 将特征值由大到小排序，存入eigvalue
            eigvalue = list(np.array(eigvalue,dtype=float))
            eigvalue.sort(reverse=True)
            # 计算每个特征值的方差贡献率，存入varcontribution 中
            varcontribution = list(np.array(eigvalue/sum(eigvalue),dtype=float))
            # 累计方差贡献率
            leiji_varcontribution = []
            for i in range(len(varcontribution)):
                s = float(sum(varcontribution[:i+1]))
                leiji_varcontribution.append(s)
            # 将特征值、方差贡献率、累计方差贡献率写入DataFrame
            # 控制列的输出顺序
            col = ['Eigvalue','Proportion','Cumulative']
            eig_df = pd.DataFrame({'Eigvalue':eigvalue,'Proportion':varcontribution,'Cumulative':leiji_varcontribution},columns=col)

            '''
            碎石图和带有方差贡献率的碎石图，此图需输出
            '''
            self.eigvalue = eigvalue
            self.eigvector = eigvector
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(6.8, 3)
            fig.subplots_adjust(wspace=0.4)
            ax1.plot(range(1, len(eig_df) + 1), eig_df['Eigvalue'], 'o-')
            ax1.set_title('Scree Plot')
            ax1.set_xlabel('Principal Components')
            ax1.set_ylabel('Eigenvalue')
            ax1.grid()

            ax2.plot(range(1, len(eig_df) + 1), eig_df['Proportion'], 'o-')
            ax2.plot(range(1, len(eig_df) + 1), eig_df['Cumulative'], 'bo-.')
            ax2.set_title('Variance Explained')
            ax2.set_xlabel('Principal Components')
            ax2.set_ylabel('Proportion')
            ax2.grid()
            plt.show()

            Eig_contri = eig_df
            Eig_contri['Eigvalue'] = Eig_contri['Eigvalue'].apply(lambda x: format(x, '.4f'))
            Eig_contri['Proportion'] = Eig_contri['Proportion'].apply(lambda x: format(x, '.2%'))
            Eig_contri['Cumulative'] = Eig_contri['Cumulative'].apply(lambda x: format(x, '.2%'))
            res = []
            res.append({
                'title': "总方差解释",
                'col': ['特征值', '特征值方差贡献率', '累计方差贡献率'],
                'data': Eig_contri.values.tolist()
            })
            res.append({
                "title": "碎石图",
                "base64": "{}".format(plot_and_output_base64_png(plt))
            })
            return res
        elif self.standardize == False:
            z = data
            # 按列求解相关系数矩阵，存入cor中（rowvar = 0指定按列求解相关系数矩阵）
            cor = np.corrcoef(z, rowvar=0)
            # 求解相关系数矩阵的特征值与特征向量，并按照特征值由大到小排序
            # 注意numpy 中求出的特征向量是按列排序而非行，因此注意将矩阵装置
            eigvalue, eigvector = np.linalg.eig(cor)
            eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue, dtype=float, columns=['eigvalue']))
            # 将特征值向量按特征值由大到小排序
            eigdf = eigdf.sort_values('eigvalue')[::-1]
            # 将调整好的特征向量存储在eigvalue
            eigvector = np.array(eigdf.iloc[:, :-1])
            # 将特征值由大到小排序，存入eigvalue
            eigvalue = list(np.array(eigvalue, dtype=float))
            eigvalue.sort(reverse=True)
            # 计算每个特征值的方差贡献率，存入varcontribution 中
            varcontribution = list(np.array(eigvalue / sum(eigvalue), dtype=float))
            # 累计方差贡献率
            leiji_varcontribution = []
            for i in range(len(varcontribution)):
                s = float(sum(varcontribution[:i + 1]))
                leiji_varcontribution.append(s)
            # 将特征值、方差贡献率、累计方差贡献率写入DataFrame
            # 控制列的输出顺序
            col = ['Eigvalue', 'Proportion', 'Cumulative']
            eig_df = pd.DataFrame(
                {'Eigvalue': eigvalue, 'Proportion': varcontribution, 'Cumulative': leiji_varcontribution}, columns=col)

            '''
            碎石图和带有方差贡献率的碎石图，此图需输出
            '''
            self.eigvalue = eigvalue
            self.eigvector = eigvector
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(6.8, 3)
            fig.subplots_adjust(wspace=0.4)
            ax1.plot(range(1, len(eig_df) + 1), eig_df['Eigvalue'], 'o-')
            ax1.set_title('Scree Plot')
            ax1.set_xlabel('Principal Components')
            ax1.set_ylabel('Eigenvalue')
            ax1.grid()

            ax2.plot(range(1, len(eig_df) + 1), eig_df['Proportion'], 'o-')
            ax2.plot(range(1, len(eig_df) + 1), eig_df['Cumulative'], 'bo-.')
            ax2.set_title('Variance Explained')
            ax2.set_xlabel('Principal Components')
            ax2.set_ylabel('Proportion')
            ax2.grid()
            plt.show()

            Eig_contri = eig_df
            Eig_contri['Eigvalue'] = Eig_contri['Eigvalue'].apply(lambda x: format(x, '.4f'))
            Eig_contri['Proportion'] = Eig_contri['Proportion'].apply(lambda x: format(x, '.2%'))
            Eig_contri['Cumulative'] = Eig_contri['Cumulative'].apply(lambda x: format(x, '.2%'))
            res = []
            res.append({
                'title': "总方差解释",
                'col': ['特征值', '特征值方差贡献率', '累计方差贡献率'],
                'data': Eig_contri.values.tolist()
            })
            res.append({
                "title": "碎石图",
                "base64": "{}".format(plot_and_output_base64_png(plt))
            })
            return res
    def var_contri(self, data):
            data = data.astype(float)
            '''
            该方法用于输出特征值、方差贡献率以及累计方差贡献率
            '''
            # 将数据转存为数组形式方便操作
            var_name = data.columns
            data = np.array(data)
            if self.standardize == True:
                z = FA.standardization(self,data)
                # 按列求解相关系数矩阵，存入cor中（rowvar = 0指定按列求解相关系数矩阵）
                cor = np.corrcoef(z, rowvar=0)
                # 求解相关系数矩阵的特征值与特征向量，并按照特征值由大到小排序
                # 注意numpy 中求出的特征向量是按列排序而非行，因此注意将矩阵装置
                eigvalue, eigvector = np.linalg.eig(cor)
                eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue, dtype=float, columns=['eigvalue']))
                # 将特征值向量按特征值由大到小排序
                eigdf = eigdf.sort_values('eigvalue')[::-1]
                # 将调整好的特征向量存储在eigvalue
                eigvector = np.array(eigdf.iloc[:, :-1])
                # 将特征值由大到小排序，存入eigvalue
                eigvalue = list(np.array(eigvalue, dtype=float))
                eigvalue.sort(reverse=True)
                # 计算每个特征值的方差贡献率，存入varcontribution 中
                varcontribution = list(np.array(eigvalue / sum(eigvalue), dtype=float))
                # 累计方差贡献率
                leiji_varcontribution = []
                for i in range(len(varcontribution)):
                    s = float(sum(varcontribution[:i + 1]))
                    leiji_varcontribution.append(s)
                # 将特征值、方差贡献率、累计方差贡献率写入DataFrame
                # 控制列的输出顺序
                col = ['Eigvalue', 'Proportion', 'Cumulative']
                eig_df = pd.DataFrame(
                    {'Eigvalue': eigvalue, 'Proportion': varcontribution, 'Cumulative': leiji_varcontribution}, columns=col)
                self.eigvalue = eigvalue
                self.eigvector = eigvector
                return eig_df
            elif self.standardize == False:
                z = data
                # 按列求解相关系数矩阵，存入cor中（rowvar = 0指定按列求解相关系数矩阵）
                cor = np.corrcoef(z, rowvar=0)
                # 求解相关系数矩阵的特征值与特征向量，并按照特征值由大到小排序
                # 注意numpy 中求出的特征向量是按列排序而非行，因此注意将矩阵装置
                eigvalue, eigvector = np.linalg.eig(cor)
                eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue, dtype=float, columns=['eigvalue']))
                # 将特征值向量按特征值由大到小排序
                eigdf = eigdf.sort_values('eigvalue')[::-1]
                # 将调整好的特征向量存储在eigvalue
                eigvector = np.array(eigdf.iloc[:, :-1])
                # 将特征值由大到小排序，存入eigvalue
                eigvalue = list(np.array(eigvalue, dtype=float))
                eigvalue.sort(reverse=True)
                # 计算每个特征值的方差贡献率，存入varcontribution 中
                varcontribution = list(np.array(eigvalue / sum(eigvalue), dtype=float))
                # 累计方差贡献率
                leiji_varcontribution = []
                for i in range(len(varcontribution)):
                    s = float(sum(varcontribution[:i + 1]))
                    leiji_varcontribution.append(s)
                # 将特征值、方差贡献率、累计方差贡献率写入DataFrame
                # 控制列的输出顺序
                col = ['Eigvalue', 'Proportion', 'Cumulative']
                eig_df = pd.DataFrame(
                    {'Eigvalue': eigvalue, 'Proportion': varcontribution, 'Cumulative': leiji_varcontribution},
                    columns=col)
                self.eigvalue = eigvalue
                self.eigvector = eigvector
                return eig_df
    def loadings(self,data):
        data = data.astype(float)
        '''
        该方法用于输出旋转前的因子载荷矩阵
        '''
        factor_num = self.component
        # 接下来求解因子载荷矩阵
        # 生成由前factor_num个特征值构成的对角阵，存入duijiao中用于计算因子载荷矩阵
        eigvalue = self.var_contri(data)['Eigvalue'] ##
        duijiao = list(np.array(np.sqrt(eigvalue[:factor_num]),dtype=float))
        eigmat = np.diag(duijiao)
        zaihe = np.dot(self.eigvector[:factor_num].T,eigmat)
        self.zaihe = zaihe
        n = range(1,factor_num+1)
        col = []
        for i in n:
            c = 'Factor'+str(i)
            col.append(c)
        zaihe = -pd.DataFrame(zaihe,columns=col)
        zaihe.iloc[:,1] = -zaihe.iloc[:,1]
        self.col = col
        zaihe.index = data.columns
        self.zaihe = zaihe
        self.zaihe = format_data_col(self.zaihe)
        col = self.zaihe.columns.values.tolist()
        row = self.zaihe.index.values.tolist()
        res = self.zaihe.values.tolist()
        return transform_table_data_to_html({
            'title': "旋转前因子载荷",
            'col': col,
            'row': row,
            'data': res
        })
    def load(self,data):
        data = data.astype(float)
        '''
        该方法用于输出旋转前的因子载荷矩阵
        '''
        factor_num = self.component
        # 接下来求解因子载荷矩阵
        # 生成由前factor_num个特征值构成的对角阵，存入duijiao中用于计算因子载荷矩阵
        eigvalue = self.var_contri(data)['Eigvalue'] ##
        duijiao = list(np.array(np.sqrt(eigvalue[:factor_num]),dtype=float))
        eigmat = np.diag(duijiao)
        zaihe = np.dot(self.eigvector[:factor_num].T,eigmat)
        self.zaihe = zaihe
        n = range(1,factor_num+1)
        col = []
        for i in n:
            c = 'Factor'+str(i)
            col.append(c)
        zaihe = -pd.DataFrame(zaihe,columns=col)
        zaihe.iloc[:,1] = -zaihe.iloc[:,1]
        self.col = col
        zaihe.index = data.columns
        self.zaihe = zaihe
        return zaihe
    def varimax_rotation(self,data):
        data = data.astype(float)
        '''
        该方法对因子载荷矩阵进行最大方差正交矩阵，返回旋转后的因子载荷矩阵
        '''
        zaihe = self.load(data)
        m,n = zaihe.shape
        R = np.eye(n)
        d = 0
        for i in range(self.q):
            d_init = d
            Lambda = np.dot(zaihe,R)
            w,a,wa = np.linalg.svd(np.dot(zaihe.T, np.asarray(Lambda)**3 - (self.gamma/m)*np.dot(Lambda,np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(w,wa)
            d = np.sum(a)
            if d_init != 0 and d/d_init < 1+self.tol:
                break
        orthogonal = np.dot(zaihe,R)
        self.orthogonal = orthogonal
        after = pd.DataFrame(orthogonal,index=data.columns,columns=self.col)
        after = format_data_col(after)
        col = after.columns.values.tolist()
        row = after.index.values.tolist()
        res = after.values.tolist()
        return transform_table_data_to_html({
            'title': "旋转后因子载荷",
            'col': col,
            'row': row,
            'data': res
        })
    def varimax_rota(self,data):
        data = data.astype(float)
        '''
        该方法对因子载荷矩阵进行最大方差正交矩阵，返回旋转后的因子载荷矩阵
        '''
        zaihe = self.load(data)
        m,n = zaihe.shape
        R = np.eye(n)
        d = 0
        for i in range(self.q):
            d_init = d
            Lambda = np.dot(zaihe,R)
            w,a,wa = np.linalg.svd(np.dot(zaihe.T, np.asarray(Lambda)**3 - (self.gamma/m)*np.dot(Lambda,np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(w,wa)
            d = np.sum(a)
            if d_init != 0 and d/d_init < 1+self.tol:
                break
        orthogonal = np.dot(zaihe,R)
        self.orthogonal = orthogonal
        after = pd.DataFrame(orthogonal,index=data.columns,columns=self.col)
        return after

    def score_coef(self,data):
        data = data.astype(float)
        '''
        该方法用于计算因子得分函数
        '''
        # R 为原始变量的相关矩阵
        corr = np.corrcoef(data,rowvar=0)
        A = self.varimax_rota(data)
        coefficient = pd.DataFrame(np.dot(np.array(A).T,np.mat(corr).T),columns=data.columns,index=self.col)
        self.coefficient = coefficient
        defen = coefficient.T
        defen = format_data_col(defen)
        col = defen.columns.values.tolist()
        row = defen.index.values.tolist()
        res = defen.values.tolist()
        return transform_table_data_to_html({
            'title': "因子得分系数矩阵",
            'col': col,
            'row': row,
            'data': res
        })

    def score(self,data):
        data = data.astype(float)
        '''
        该方法用于计算因子得分
        '''
        if self.standardize == True:
            data_scale = FA.standardization(self,data)
            F = np.dot(data_scale,self.coefficient.T)
            F = pd.DataFrame(F)
            col2 = []
            n = range(1,self.component+1)
            for i in n:
                c  = 'ScoreF'+str(i)
                col2.append(c)
            F.columns = col2
            F = format_data_col(F)
            col = F.columns.values.tolist()
            row = F.index.values.tolist()
            res = F.values.tolist()
            return transform_table_data_to_html({
                'title': "因子得分",
                'col': col,
                'row': row,
                'data': res
            })
        elif self.standardize == False:
            data_scale = data
            F = np.dot(data_scale, self.coefficient.T)
            F = pd.DataFrame(F)
            col2 = []
            n = range(1, self.component + 1)
            for i in n:
                c = 'ScoreF' + str(i)
                col2.append(c)
            F.columns = col2
            F = format_data_col(F)
            col = F.columns.values.tolist()
            row = F.index.values.tolist()
            res = F.values.tolist()
            return transform_table_data_to_html({
                'title': "因子得分",
                'col': col,
                'row': row,
                'data': res
            })

if __name__ == '__main__':
    # new_data = data_standard(data)
    import os
    os.chdir('/Users/chuckzhao/Documents/qwf/pyworkspace/tool_data')
    data = pd.read_excel('principal.xlsx')
    ic_fa = FA(component=3,standardize=True,transpose=True)
    bar = ic_fa.kmo_Bartlett(data)
    print(bar)
    contribution = ic_fa.var_contribution(data) # 特征值及贡献率及碎石图
    print(contribution)
    before_zaihe = ic_fa.loadings(data) # 旋转前载荷矩阵
    print(before_zaihe)
    after_zaihe = ic_fa.varimax_rotation(data) # 旋转后载荷矩阵
    print(after_zaihe)
    score_coef = ic_fa.score_coef(data) # 因子得分系数
    print(score_coef)
    score = ic_fa.score(data) # 因子得分
    print(score)

