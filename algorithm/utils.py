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
import logging
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus
import base64
from io import BytesIO

log = logging.getLogger(__name__)


# ======================= 算法预处理 =============================
# 特征编码
def data_encoder(data: pd.DataFrame, column_name_list, use_onehot=False, default_value=0):
    for column_name in column_name_list:
        if use_onehot:
            if column_name not in data.columns:
                raise ValueError("{} not in {} columns".format(column_name, data))
            one_data = pd.get_dummies(data[column_name])
            one_data.columns = [column_name + "_" + c for c in one_data.columns]
            data = data.join(one_data)
            data.drop([column_name], axis=1, inplace=True)
        else:
            # Replace missing values with "default_value"
            data[column_name][data[column_name].isnull()] = default_value
            # convert the distinct cabin letters with incremental integer values
            data[column_name] = pd.factorize(data[column_name])[0]
    return data


# 归一化
def data_standard(data: pd.DataFrame, column_name_list, method="normal"):
    try:
        data[column_name_list] = data[column_name_list].astype("float")
    except ValueError as e:
        log.exception("data_standard_error")
        raise e
    for column_name in column_name_list:
        if method == "minMaxScale":
            data[column_name] = (data[column_name] - data[column_name].min()) / (
                    data[column_name].max() - data[column_name].min())
        if method == "standard":
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


def get_dataframe_from_mysql(sql_sentence, host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com',
                             port=3306, user='yzkj', password='yzkj2020@',
                             database='sophia_manager', charset='utf8'):
    conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database, charset=charset)
    try:
        df = pd.read_sql(sql_sentence, conn)
        return df
    except Exception as e:
        raise e


# 根据sql获取数据
def exec_sql(table_name, X=None, Y=None):
    # 从数据库拿数据
    try:
        if not Y or Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), "`" + table_name + "`")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), "`" + table_name + "`")
        data = get_dataframe_from_mysql(sql_sentence)
        return data
    except Exception as e:
        log.info(e.args)
        raise e


# 根据算法查询模型
def exec_select_sql(sql, host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com',
                    port=3306, user='yzkj', password='yzkj2020@',
                    database='sophia_manager', charset='utf8'):
    conn = pymysql.connect(host=host, port=port, user=user,
                           password=password, database=database, charset=charset)
    cursor = conn.cursor()
    res = []
    try:
        # Execute the SQL command
        cursor.execute(sql)
        res = cursor.fetchall()
        # Commit your changes in the database
        conn.commit()
    except Exception as e:
        log.error(e)
        # Rollback in case there is any error
        conn.rollback()
    conn.close()
    return res


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


# 分类评估报告转表格数据输出给前端
def report_to_table_data(report):
    col = ["精确率", "召回率", "F1", "样本数"]
    row = []
    data = []
    for row_data in report.split("\n\n")[1].split("\n"):
        row_data = [r for r in row_data.split(" ") if r]
        row.append(row_data[0])
        data.append(row_data[1:])
    for row_data in report.split("\n\n")[2].split("\n"):
        if not row_data:
            continue
        row_data = [r for r in row_data.split(" ") if r]
        if row_data[0] == "accuracy":
            row.append(row_data[0])
            data.append(["", ""] + row_data[1:])
        else:
            row.append(row_data[0] + " " + row_data[1])
            data.append(row_data[2:])
    return {
        "row": row,
        "col": col,
        "data": data,
        "title": "分类报告:precision/recall/F1/分类个数",
        "remarks": "accuracy:准确率(正负样本总的正确分类的比率)，"
                   "macro avg:宏平均(所有类的精确率、召回率、F1的平均值), "
                   "weighted avg:加权平均基于样本个数加权平均精确率、召回率、F1)"
    }


# 机器学习模型分类效果展示
def show_classifier_results(x, y, model, options=[]):
    """
    机器学习模型分类效果展示
    :param x: 特征列
    :param y: 标签列
    :param model: 已经训练好的分类模型
    :param options: 可选参数，控制输出结果["report", "matrix", "roc"]
    :return: 给前端的结果
    """
    res = []
    # 输出结果展示
    y_predict = model.predict(x)
    y_predict_proba = model.predict_proba(x)

    # 分类评估报告输出表格数据,默认展示
    # accuracy_score = metrics.accuracy_score(y, y_predict)
    # precision_score = metrics.precision_score(y, y_predict)
    # recall_score = metrics.recall_score(y, y_predict)
    # f1_score = metrics.f1_score(y, y_predict)
    report = metrics.classification_report(y, y_predict, target_names=model.classes_.tolist())
    res.append(transform_table_data_to_html(report_to_table_data(report)))

    # 输出混淆矩阵图片
    if "matrix" in options:
        metrics.plot_confusion_matrix(model, x, y)
        res.append({
            "title": "混淆矩阵",
            "base64": "{}".format(plot_and_output_base64_png(plt))
        })

    # 输出roc、auc图片
    if "roc" in options:
        metrics.plot_roc_curve(model, x, y)
        res.append({
            "title": "ROC曲线和auc",
            "base64": "{}".format(plot_and_output_base64_png(plt))
        })

    return res


# 决策树模型可视化
def generate_tree_graph(model, feature_names, class_names):
    """
    可视化决策树图
    :param model: 决策树模型
    :param feature_names: 特征名列表
    :param class_names: 标签名列表
    :return: 图
    """
    #
    # dot_data = StringIO()
    dot_data = export_graphviz(model,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True,
                               rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    img = graph.create_png()
    return


# matplotlib作图写入内存并输出base64格式供前端调用
def plot_and_output_base64_png(plot):
    # 写入内存
    save_file = BytesIO()
    plot.savefig(save_file, format='png')

    # 转换base64并以utf8格式输出
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    return save_file_base64


# 算法输出结果
def algorithm_show_result(model, x, y, options=[], method="classifier"):
    res = []
    if method == "classifier":
        # 分类测试集结果
        res = show_classifier_results(x, y, model, options=options)

    if method == "regression":
        # 拟合优度结果（回归算法才有）
        try:
            import statsmodels.api as sm
        except:
            raise ImportError("statsmodels.api cannot import")
        try:
            x = sm.add_constant(x)
            logit_stats_res = sm.Logit(y, x).fit()
            # 拟合优度
            if "r2" in options:
                res.append({
                    "title": "逻辑回归统计分析结果",
                    "data": str(logit_stats_res.fit().summary().tables[0])
                })
            # 系数解读
            if "coff" in options:
                res.append({
                    "title": "逻辑回归系数解读",
                    "data": str(logit_stats_res.fit().summary().tables[1])
                })
        except Exception as e:
            log.error("statsmodels analysis error")
            # raise e
    if method == "cluster":
        pass
    return res


# 自变量根据用户指定的最高阶数生成新的数据
def gen_poly_col(data, conf):
    """

    :param data:
    :param conf: {"x1": 3, "x2": 4}
    :return:
    """
    for x in conf:
        if not isinstance(conf[x], int):
            conf[x] = int(conf[x])
        for i in range(2, conf[x] + 1):
            data["{}**{}".format(x, i)] = data[x] ** i
    return data


# 保存模型接口，实行插入sql语句
def exec_insert_sql(table, key_list, value_list, host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com',
                    port=3306, user='yzkj', password='yzkj2020@',
                    database='sophia_manager', charset='utf8'):
    conn = pymysql.connect(host=host, port=port, user=user,
                           password=password, database=database, charset=charset)
    cursor = conn.cursor()
    sql_list = []
    sql = "INSERT INTO {}({}) VALUES ('{}')".format(table, ",".join(key_list), "','".join(value_list))
    sql_list.append(sql)
    try:
        # Execute the SQL command
        cursor.execute(sql)
        # Commit your changes in the database
        conn.commit()
    except Exception as e:
        log.error(e)
        # Rollback in case there is any error
        conn.rollback()
    conn.close()
    return sql_list
