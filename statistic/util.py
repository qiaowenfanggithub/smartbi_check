# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : utils

Description : 

Author : leiliang

Date : 2020/7/9 3:54 下午

--------------------------------------------------------

"""
import base64
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import logging

log = logging.getLogger(__name__)


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
    return data.std() / data.mean()


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


# 根据sql获取数据
def exec_sql(table_name, X=None, Y=None):
    # 从数据库拿数据
    try:
        if not X and not Y:
            sql_sentence = "select * from {};".format("`" + table_name + "`")
        elif not Y or Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), "`" + table_name + "`")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), "`" + table_name + "`")
        data = get_dataframe_from_mysql(sql_sentence)
        return data
    except Exception as e:
        log.info(e.args)
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


# 当是一列分类变量，一列数值型变量，将各个分类对应的数值型变量自成一列，由【分类变量，数值型变量】变成【数值型变量1，数值型变量2，...】
def transform_v_table_data_to_h(data: pd.DataFrame, X, Y):
    list = []
    col = [d for d in data[X[0]].unique()]
    for i in col:
        zh = data[data[X[0]] == i]
        l = zh.iloc[:, -1].tolist()
        list.append(l)
    new_data = pd.DataFrame(list).T
    new_data.columns = col
    X = col
    return new_data, X


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


# 对无列名的dataframe，将数据全转成整数型
def format_data(data):
    data = data.astype(float)
    length = data.shape[1]
    for i in range(length):
        data[i] = data[i].apply(lambda x: "{:.0f}".format(x))
    return data

# 对有列名的dataframe，将数据全转成4位小数
def format_data_col(data):
    data = data.astype(float)
    length = data.shape[1]
    for i in range(length):
        data.iloc[0:,i] = data.iloc[0:,i].apply(lambda x: "{:.4f}".format(x))
    return data

#  matplotlib作图写入内存并输出base64格式供前端调用
def plot_and_output_base64_png(plot):
    plot.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plot.rcParams["axes.unicode_minus"] = False
    # 写入内存
    save_file = BytesIO()
    plot.savefig(save_file, format='png')
    # 转换base64并以utf8格式输出
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    # debug
    # base64_to_img(save_file_base64)
    plot.close("all")

    # 写入文件
    # tmp_file_name = uuid.uuid4()
    # plot.savefig("./img/{}.png".format(tmp_file_name))
    # with open("./img/{}.png".format(tmp_file_name), "rb") as f:
    #     save_file_base64 = base64.b64encode(f.read()).decode('utf8')

    return save_file_base64
