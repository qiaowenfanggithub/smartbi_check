# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : flask_test

Description : 

Author : leiliang

Date : 2020/6/28 4:41 下午

--------------------------------------------------------

"""
from __future__ import print_function
import requests

if __name__ == '__main__':
    my_session = requests.session()
    # ======================= 决策树-训练 =============================
    # kwargs = {
    #     "isTrain": False,  # str,数据库表名
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,特征
    #     "Y": ["是否购买电脑"],  # list,标签
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "2",  # str,几折交叉验证
    #     "param": {
    #         "criterion": ["gini"],  # 不纯度指标gini、entropy
    #         "max_features": ["3", "4"],
    #         "max_depth": ["2", "3"],  # 指定树的最大深度
    #         "min_samples_split": ["2"],  # :int, float, optional (default=2)。表示分裂一个内部节点需要的最少样本数。
    #         "min_samples_leaf": ["1", "2"],  # int, float, optional (default=1)。指定每个叶子节点需要的最少样本数。
    #     },
    #     "show_options": [
    #             "report",
    #             "matrix",
    #             "roc"
    #         ]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/train', json=kwargs, timeout=30)

    # ======================= 决策树-预测(多个测试样本) =============================
    # kwargs = {
    #     "algorithm": "decisionTree",  # str,数据库表名
    #     "model": "decisionTree-2020-08-04-22-39-36",  # str,数据库表名
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    #     "show_options": [
    #                 "report",
    #                 "matrix",
    #                 "roc"
    #             ]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/predict', json=kwargs, timeout=30)

    # ======================= 决策树-预测(单个测试样本) =============================
    # kwargs = {
    #     "algorithm": "decisionTree",  # str,数据库表名
    #     "model": "decisionTree-2020-08-04-22-39-36",  # str,数据库表名
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     "X": [0, 0, 0, 0],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/predict', json=kwargs, timeout=30)

    # ======================= 决策树-评估 =============================
    # kwargs = {
    #     "algorithm": "decisionTree",  # str,数据库表名
    #     "model": "decisionTree-2020-08-04-22-39-36",  # str,数据库表名
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    #     "Y": ["是否购买电脑"],  # list,标签
    #     "show_options": [
    #                 "report",
    #                 "matrix",
    #                 "roc"
    #             ]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/evaluate', json=kwargs, timeout=30)

    # ======================= 逻辑回归-训练 =============================
    # kwargs = {
    #     "tableName": "91ceb15911c0441e86eeb791a6d08720",
    #     "X": [
    #         "年龄",
    #         "收入层次",
    #         "是否单身",
    #         "信用等级"
    #     ],
    #     "Y": [
    #         "是否购买电脑"
    #     ],
    #     "rate": "0.4",
    #     "randomState": "2",
    #     "cv": "2",
    #     "param": {
    #         "penalty": [
    #             "l1"
    #         ],
    #         "C": [
    #             "1"
    #         ],
    #         "solver": [
    #             "liblinear",
    #             "saga"
    #         ],
    #         "max_iter": [
    #             "100"
    #         ],
    #         "fit_intercept": [
    #             True
    #         ]
    #     },
    #     "show_options": [
    #         "report",
    #         "matrix",
    #         "roc",
    #         "r2",
    #         "coff",
    #         "independence",
    #         "resid_normal",
    #         "pp",
    #         "qq",
    #         "var",
    #         "vif",
    #         "outliers",
    #         "pred_y_contrast"
    #     ]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/logisticRegression/train', json=kwargs, timeout=30)

    # ======================= 逻辑回归-评估 =============================
    # kwargs = {
    #     "algorithm": "logisticRegression",  # str,数据库表名
    #     "model": "logisticRegression-2020-08-04-22-15-36",  # str,数据库表名
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "show_options": ["matrix", "roc", "coff", "independence"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/logisticRegression/evaluate', json=kwargs, timeout=30)

    # ======================= 逻辑回归-预测 =============================
    # kwargs = {
    #     "algorithm": "logisticRegression",  # str,数据库表名
    #     "model": "logisticRegression-2020-08-04-22-15-36",  # str,数据库表名
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     # "X": [1, 1, 1, 0],  # list,自变量
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/logisticRegression/predict', json=kwargs, timeout=30)

    # ======================= 线性回归-训练 =============================
    kwargs = {
        "tableName": "liner_regression",  # str,数据库表名
        "X": ["year"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["salary"],  # list,因变量,当表格方向为v是使用
        "param": {"fit_intercept": True},  # bool,True或者False，是否有截距项
        "show_options": ["r2", "coff", "Independence", "resid_normal",
                         "pp", "qq", "var", "vif", "outliers", "pred_y_contrast"]
    }
    res = my_session.post(url='http://127.0.0.1:5000/algorithm/linerRegression/train', json=kwargs, timeout=30)

    # ======================= 线性回归-预测 =============================
    # kwargs = {
    #     "algorithm": "linerRegression",  # str,数据库表名
    #     "model": "linerRegression-2020-08-04-22-43-24",  # str,数据库表名
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     # "tableName": "liner_regression",  # str,数据库表名
    #     "X": [12.0],  # list,自变量
    #     # "X": ["year"],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/linerRegression/predict', json=kwargs, timeout=30)

    # ======================= 线性回归-可视化 =============================
    # kwargs = {
    #     "tableName": "advertising",  # str,数据库表名
    #     "X": ["TV", "radio", "newspaper"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["sales"],  # list,因变量,当表格方向为v是使用
    #     "show_options": ["y_count", "pairs", "corr", "y_corr"],
    #     "x_count": ["TV", "radio", "newspaper"],
    #     "box": ["TV", "radio", "newspaper"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/linerRegression/visualization', json=kwargs, timeout=30)

    # ======================= 多项回归-训练 =============================
    # kwargs = {
    #     "tableName": "poly_reg",  # str,数据库表名
    #     "X": ["x1", 'x2', 'x3'],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["y"],  # list,因变量,当表格方向为v是使用
    #     "M": [{'x1': "2", 'x2': "2", 'x3': "2"}],
    #     "param": {"fit_intercept": True},  # bool,True或者False，是否有截距项
    #     "show_options": ["r2", "coff", "Independence", "resid_normal",
    #                      "pp", "qq", "var", "vif", "outliers", "pred_y_contrast"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/polyLinerRegression/train', json=kwargs, timeout=30)

    # ======================= 多项回归-预测=============================
    # kwargs = {
    #     "algorithm": "polyLinerRegression",  # str,数据库表名
    #     "model": "polyLinerRegression-2020-08-04-22-54-25",  # str,数据库表名
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     # "tableName": "poly_regression",  # str,数据库表名
    #     "X": [18, 2, 2],  # list,自变量
    #     # "X": ["x1", 'x2', 'x3'],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/polyLinerRegression/predict', json=kwargs, timeout=30)

    # ======================= K-Means 聚类-训练 =============================
    # kwargs = {
    #     "tableName": "iris_kmeans",  # str,数据库表名
    #     # "X": ["sl", "sw"],  # list,自变量，每个元素是浮点类型
    #     "X": ["sl", "sw", "pl", "pw"],  # list,自变量，每个元素是浮点类型
    #     "param": {
    #         "n_clusters": "3",  # list,聚类中心数量，默认2个,如果是多个画图展示每个聚类的效果
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/kMeans/train', json=kwargs, timeout=30)

    # ======================= K-Means 聚类-预测 =============================
    # kwargs = {
    #     "algorithm": "kmeans",  # str,数据库表名
    #     "model": "kmeans-2020-08-04-23-01-42",  # str,数据库表名
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "iris_kmeans",  # str,数据库表名
    #     # "X": ["sl", "sw"],  # list,自变量，每个元素是浮点类型
    #     "X": ["sl", "sw", "pl", "pw"],  # list,自变量，每个元素是浮点类型
    #     # "X": ["5.1", "3.5", "1.4", "0.2"],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/kMeans/predict', json=kwargs, timeout=30)

    # ======================= 随机森林-训练 =============================
    # kwargs = {
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "3",  # str,几折交叉验证
    #     "param": {
    #         "n_estimators": ["100", "200"],  # list,树的个数
    #         "criterion": ["gini"],  # list,树划分准则
    #         "max_features": ["2", "3"],  # list，用于训练的最大特征数量
    #         "max_depth": ["5", "6"],  # list，树的最大深度列表
    #         "min_samples_split": ["2", "3"],  # list， 内部节点再划分所需最小样本数
    #         "min_samples_leaf": ["1", "2"],  # list，叶子节点最少样本数
    #     },
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/train', json=kwargs, timeout=50)

    # ======================= 随机森林-评估 =============================
    # kwargs = {
    #     "algorithm": "randomForest",  # str,数据库表名
    #     "model": "randomForest-2020-08-04-23-03-45",  # str,数据库表名
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/evaluate', json=kwargs, timeout=50)

    # ======================= 随机森林-预测 =============================
    # kwargs = {
    #     "algorithm": "randomForest",  # str,数据库表名
    #     "model": "randomForest-2020-08-04-23-03-45",  # str,数据库表名
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     # "X": [0, 1, 1, 1],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/predict', json=kwargs, timeout=50)

    # ======================= 评估-总入口 =============================
    # kwargs = {
    #     "algorithm": "logisticRegression",
    #     "model": "logisticRegression-2020-08-06-15-58-29",
    #     "tableName": "buy_computer_new",
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],
    #     "Y": ["是否购买电脑"],
    #     "show_options": ["report", "matrix", "roc", "r2", "coff", "independence"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/evaluate', json=kwargs, timeout=50)

    # ======================= 预测-总入口 =============================
    # kwargs = {
    #     "algorithm": "randomForest",
    #     "model": "randomForest-2020-08-04-23-03-45",
    #     "oneSample": False,
    #     "tableName": "buy_computer_new",
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],
    #     # "X": [0, 1, 1, 1]
    #
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/predict', json=kwargs, timeout=50)

    # ======================= 模型查询 =============================
    # res = my_session.get(url='http://127.0.0.1:5000/algorithm/selectModel/logisticRegression', timeout=50)

    # ======================= 模型特征查看接口 =============================
    # kwargs = {
    #     "algorithm": "logisticRegression",
    #     "model": "logisticRegression-2020-08-06-15-58-29",
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/checkModelFeatures', json=kwargs, timeout=50)

    # ======================= 数据预处理-编码 =============================
    # kwargs = {
    #     "tableName": "buy-computer",  # str,数据库表名
    #     "encoder": {
    #         "oneHot": ["年龄", "收入层次"],  # list,需要使用onehot编码的特征列
    #         "factorize": ["是否单身", "信用等级", "是否购买电脑"]  # list,需要使用序列编码的特征列
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/dataProcess/encoder', json=kwargs, timeout=50)

    # ======================= 数据预处理-归一化 =============================
    # kwargs = {
    #     "tableName": "advertising",  # str,数据库表名
    #     "normalize": {
    #         "minMaxScale": [],  # list,需要使用normal标准化的特征列
    #         "standard": ["TV", "radio", "newspaper"]  # list,需要使用归一化的特征列
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/dataProcess/normalize', json=kwargs, timeout=50)

    """
    =====================================================================
                                    待完成
    =====================================================================
    """

    # ======================= 支持向量机 =============================
    # ======================= 层次聚类 ==============================
    print(res.text)
