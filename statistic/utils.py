# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : utils

Description : 

Author : leiliang

Date : 2020/7/9 3:54 下午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql


# ======================= 算法预处理 =============================
# 特征编码
def data_encoder(data: pd.DataFrame, column_name_list, use_onehot=False, default_value=0):
    for column_name in column_name_list:
        if use_onehot:
            if column_name not in data.columns:
                raise ValueError("{} not in {} columns".format(column_name, data))
            data = data.join(pd.get_dummies(data[column_name]))
            data.drop([column_name], axis=1, inplace=True)
        else:
            # Replace missing values with "default_value"
            data[column_name][data[column_name].isnull()] = default_value
            # convert the distinct cabin letters with incremental integer values
            data[column_name] = pd.factorize(data[column_name])[0]
    return data


# 归一化
def data_standard(data: pd.DataFrame, column_name_list, method="normal"):
    for column_name in column_name_list:
        if method == "normal":
            data[column_name] = (data[column_name] - data[column_name].min()) / (
                    data[column_name].max() - data[column_name].min())
        else:
            data[column_name] = (data[column_name] - data[column_name].mean()) / (data[column_name].std())
    return data


# 计算向量相似度
def data_similarity(data0, data1):
    assert len(data0) == len(data1)
    return np.round(cosine_similarity([data0, data1])[0][1], 8)


# ======================= 一般描述统计 =============================
# 计算平均值
def data_mean(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.mean()


# 计算中位数
def data_median(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.median()


# 计算众数
def data_mode(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.mode()


# 计算方差
def data_var(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.var()


# 计算标准差
def data_std(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.std()


# 计算极数
def data_extreme(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.max(), data.min()


# ======================= 数据概况 =============================
# 变异系数
def data_cv(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.mean() / data.std()


# 频数分布
def data_count(data, col_name=None):
    if col_name:
        data = data[col_name]
    return data.value_counts()


# 四分位数
def data_quantity(data, col_name=None, quantity=[0.25, 0.75]):
    if col_name:
        data = data[col_name]
    if isinstance(quantity, float):
        quantity = [quantity]
    return data.quantile(quantity)


# ======================= 探索性分析 =============================
# 频数统计（0-1直方图）
def data_count_plot(data, col_name=None, hue=None):
    sns.countplot(x=col_name, hue=hue, data=data, palette="Pastel2")
    plt.xlabel(col_name)
    plt.title("{} by {}".format(hue, col_name))
    plt.savefig("count_plot_{}_by_{}.png".format(hue, col_name))


# 相关系数矩阵
def data_corr_plot(data, figsize=(20, 16)):
    corr = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                linewidths=0.2, cmap="YlGnBu", annot=True)
    plt.title("Correlation between variables")
    plt.savefig("corr_plot.png")


# 交叉散点图（每个X与Y的散点图）
def data_scatter_plot(data, col_name_X, col_name_Y):
    sns.scatterplot(data[col_name_X], data[col_name_Y])
    plt.title("{} by {}".format(col_name_Y, col_name_X))
    plt.savefig("scatter_plot_{}_by_{}.png".format(col_name_Y, col_name_X))


def get_dataframe_from_mysql(sql_sentence, host=None, port=None, user=None, password=None, database=None):
    conn = pymysql.connect(host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com', port=3306, user='yzkj',
                           password='yzkj2020@', database='sophia_data', charset='utf8')
    try:
        df = pd.read_sql(sql_sentence, conn)
        return df
    except Exception as e:
        raise e


# 将自变量在多列的表格转成自变量在一列，因变量在一列
def transform_h_table_data_to_v(data: pd.DataFrame, X):
    level_index = []
    value = []
    for x in X:
        level_index.extend([x] * len(data[x]))
        value.extend(data[x].values.tolist())
    data = pd.DataFrame({"level": level_index, "value": value}, dtype="float16")
    X = ["level"]
    Y = ["value"]
    return data, X, Y


# 转换输出的表格数据让前端识别并显示
def transform_table_data_to_html(data: dict, col0=""):
    data["col"].insert(0, col0)
    for idx, (index, row) in enumerate(zip(data["row"], data["data"])):
        if not isinstance(data["data"][idx], list):
            data["data"][idx] = list(data["data"][idx])
        data["data"][idx].insert(0, str(index))
    if "row" in data:
        del data["row"]
    return data


# format dataframe
def format_dataframe(data, config):
    for key, value in config.items():
        data[key] = data[key].map(lambda x: format(x, value))
    return data
