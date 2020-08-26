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
import time

if __name__ == '__main__':
    my_session = requests.session()
    start_time = time.time()
    # ======================= 决策树-训练 =============================
    # kwargs = {
    #     "tableName": "iris",  # str,数据库表名
    #     # "tableName": "buy_computer_new",  # str,数据库表名
    #     # "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,特征
    #     "X": ["x0", "x1", "x2", "x3"],  # list,特征
    #     "Y": ["label"],  # list,标签
    #     # "Y": ["是否购买电脑"],  # list,标签
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "2",  # str,几折交叉验证
    #     "param": {
    #         "criterion": ["gini"],  # 不纯度指标gini、entropy
    #         "max_features": None,
    #         # "max_features": ["3", "4"],
    #         "max_depth": None,  # 指定树的最大深度
    #         # "max_depth": ["2", "3"],  # 指定树的最大深度
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
    # # 保存模型接口调试
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],
    #     "Y": ["是否购买电脑"],
    #     "rate": "0.4",
    #     "randomState": "2",
    #     "cv": "2",
    #     "param": {
    #         "penalty": ["l1"],
    #         "C": ["1"],
    #         "solver": ["liblinear", "saga"],
    #         "max_iter": ["100"],
    #         "fit_intercept": [True]
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
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res1 = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    #     model_info2 = res.json()["model_info2"]
    #     res2 = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info2, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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
    # kwargs = {
    #     "tableName": "liner_regression",  # str,数据库表名
    #     "X": ["year"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["salary"],  # list,因变量,当表格方向为v是使用
    #     "param": {"fit_intercept": True},  # bool,True或者False，是否有截距项
    #     "show_options": ["r2", "coff", "Independence", "resid_normal",
    #                      "pp", "qq", "var", "vif", "outliers", "pred_y_contrast"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/linerRegression/train', json=kwargs, timeout=30)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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
    #     },
    #     "show_options": ["cluster"]  # cluster: 聚类结果，默认是【"cluster"】
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/kMeans/train', json=kwargs, timeout=30)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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

    # ======================= 层次聚类-训练 =============================
    # kwargs = {
    #     "tableName": "iris_hie",  # str,数据库表名
    #     "X": ["x1", "x2"],  # list,自变量，每个元素是浮点类型
    #     "param": {
    #         "n_clusters": "3",  # list,聚类中心数量，默认3个,如果是多个画图展示每个聚类的效果
    #         "linkage": "ward",  # 下拉框单选，"ward"，"average"， "complete"; "ward"时只能选"euclidean"
    #         "affinity": "euclidean",  # 下拉框单选，"euclidean", "manhattan", "cosine"
    #     },
    #     "show_options": [""]  # cluster: 聚类结果，默认是【"cluster"】
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/hierarchicalCluster/train', json=kwargs, timeout=30)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

    # ======================= 层次聚类-预测 =============================
    # kwargs = {
    #     "algorithm": "hierarchicalCluster",  # str,数据库表名
    #     "model": "hierarchicalCluster-2020-08-13-11-14-21",  # str,数据库表名
    #     # 只支持批量上传数据进行预测，因为层级聚类预测最小需要3个样本
    #     "tableName": "iris_hie",  # str,数据库表名
    #     "X": ["x1", "x2"],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/hierarchicalCluster/predict', json=kwargs, timeout=30)

    # ======================= 随机森林-训练 =============================
    # kwargs = {
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "3",  # str,几折交叉验证
    #     "param": {
    #         "n_estimators": ["10", "20"],  # list,树的个数
    #         "criterion": ["gini"],  # list,树划分准则
    #         "max_features": None,  # list，用于训练的最大特征数量
    #         # "max_features": ["2", "3"],  # list，用于训练的最大特征数量
    #         "max_depth": None,  # list，树的最大深度列表
    #         # "max_depth": ["5", "6"],  # list，树的最大深度列表
    #         "min_samples_split": ["2", "3"],  # list， 内部节点再划分所需最小样本数
    #         "min_samples_leaf": ["1", "2"],  # list，叶子节点最少样本数
    #     },
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/train', json=kwargs, timeout=50)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

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

    # ======================= 支持向量机-训练 =============================
    # kwargs = {
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     # "tableName": "iris",  # str,数据库表名
    #     # "X": ["x0", "x1", "x2", "x3"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     # "Y": ["label"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "3",  # str,几折交叉验证
    #     "param": {
    #         "kernel": ["linear", "poly", "rbf", "sigmoid"],  # str,惩罚项，‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #         "C": ["2"],  # str,惩罚项系数
    #         "degree": ["3"],  # 多项式核函数的维度
    #         "gamma": ["auto"],  # ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
    #         "coef0": ["0"],  # 核函数的常数项
    #         "tol": ["0.001"],  # 停止训练的误差值大小，默认为1e-3
    #         "max_iter": ["-1"],  # 最大迭代次数。-1为无限制。
    #         "decision_function_shape": ["ovo", "ovr", ""],  # 最大迭代次数。-1为无限制。
    #     },
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/svmClassifier/train', json=kwargs, timeout=50)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

    # ======================= 多层感知机-训练 =============================
    # kwargs = {
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     # "tableName": "iris",  # str,数据库表名
    #     # "X": ["x0", "x1", "x2", "x3"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     # "Y": ["label"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "param": {
    #         "hidden_layer_sizes": ["10"],  # str（tuple）,隐藏层个数和每个隐藏层节点数
    #         "activation": "relu",  # str,激活函数["identity", "logistic", "tanh", "relu"]
    #         "solver": "adam",  # str，优化算法["lbfgs", "sgd", "adam"]
    #         "alpha": "0.0001",  # str(float)，惩罚项系数["0.0001", "0.00001"]
    #         "batch_size": "auto",  # str(int)，随机优化的minibatches的大小，默认auto，手动输入整数
    #         "learning_rate_init": "0.001",  # str(float)，初始学习率
    #         "tol": "0.0001",  # str(float)优化的容忍度
    #         "max_iter": "200",  # str(int)最大迭代次数
    #     },
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/mlpClassifier/train', json=kwargs, timeout=50)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

    # ======================= adaboost-训练 =============================
    kwargs = {
        # "tableName": "buy_computer_new",  # str,数据库表名
        # "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        # "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
        "tableName": "iris",  # str,数据库表名
        "X": ["x0", "x1", "x2", "x3"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["label"],  # list,因变量,当表格方向为v是使用
        "rate": "0.1",  # str,测试集训练集分割比例
        "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
        "cv": "10",  # str,几折交叉验证
        "learning_rate": "0.8",  # str,学习率【默认值："1"，0-1之间的小数】
        "param": {
            "criterion": ["gini", "entropy"],  # list, 树划分准则【默认值：["gini"]】
            "max_depth": ["3", "5"],  # list, 最大树深度【默认值：[none]】
            "max_features": ["3", "4"],  # list, 最大特征数【默认值：[none]】
            "min_sample_split": ["1"],  # list, 节点划分最小样本数【默认值：["2"]】
            "min_samples_leaf": ["1"],  # list, 叶子节点最小数【默认值：["1"]】
            "n_estimators": ["10", "30"],  # list,弱学习器个数【默认值：["20"]】
        },
        "show_options": ["report", "matrix", "roc"]
    }
    res = my_session.post(url='http://127.0.0.1:5000/algorithm/adaboostClassifier/train', json=kwargs, timeout=50)
    # if res.json()["code"] == "200":
    #     model_info = res.json()["model_info"]
    #     res = my_session.post(url='http://127.0.0.1:5000/algorithm/saveModel', json=model_info, timeout=30)
    # else:
    #     raise ValueError(res.json()["msg"])

    # ======================= 评估-总入口 =============================
    # kwargs = {
    #     "algorithm": "svmClassifier",
    #     "model": "svmClassifier-2020-08-11-17-13-11",
    #     "tableName": "buy_computer_new",
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],
    #     "Y": ["是否购买电脑"],
    #     "show_options": ["report", "matrix", "roc"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/evaluate', json=kwargs, timeout=50)

    # ======================= 预测-总入口 =============================
    # kwargs = {
    #     "algorithm": "svmClassifier",
    #     "model": "svmClassifier-2020-08-11-17-11-48",
    #     "oneSample": False,
    #     "tableName": "iris",
    #     "X": ["0", "1", "2", "3"],
    #     # "X": [5.1, 3.5, 1.4, 0.2]
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
    #     "tableId": "buy-computer",  # str,数据库表名
    #     "encoder": {
    #         "oneHot": ["年龄"],  # list,需要使用onehot编码的特征列
    #         "factorize": ["收入层次"]  # list,需要使用序列编码的特征列
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/dataProcess/encoder', json=kwargs, timeout=50)

    # ======================= 数据预处理-归一化 =============================
    # kwargs = {
    #     "tableId": "advertising",  # str,数据库表名
    #     "normalize": {
    #         "minMaxScale": ["TV", "radio"],  # list,需要使用normal标准化的特征列
    #         "standard": ["newspaper"]  # list,需要使用归一化的特征列
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/dataProcess/normalize', json=kwargs, timeout=50)

    # ======================= 数据探索-可视化 =============================
    # kwargs = {
    #     # "tableName": "buy-computer",  # str,数据库表名
    #     "tableName": "bankloan",  # str,数据库表名
    #     "count": ["年龄", "教育"],  # list,频率直方图字段列表
    #     # "count_hue": "违约",  # str,频率直方图分类字段
    #     "dist": ["收入"],  # list,数据分布图字段列表
    #     "box": ["工龄", "负债率"],  # list,箱型图字段列表
    #     # "pie": ["违约"],  # list,饼图字段列表
    #     # "pairPlot": ["年龄", "教育", "工龄", "地址", "收入", "负债率", "信用卡负债", "其他负债", "违约"],  # list,特征两两散点图字段列表
    #     "heatMap": ["年龄", "教育", "收入", "负债率"],  # list,相关系数热度图
    #     # "heatMap": ["年龄", "教育", "工龄", "地址", "收入", "负债率", "信用卡负债", "其他负债", "违约"],  # list,相关系数热度图
    #     # "yCorr": {
    #     #     "X": ["年龄", "教育", "工龄", "地址", "收入", "负债率", "信用卡负债", "其他负债"],
    #     #     "Y": ["违约"]
    #     # },  # list,自变量和各因变量相关系数图 ==>【分类和聚类算法变灰】
    #     "scatter": {
    #         "X": ["收入"],
    #         # "X": ["收入", "负债率", "信用卡负债"],
    #         "Y": ["其他负债"]
    #     },  # list,自变量和各因变量散点图
    #     "crosstab": {
    #         # "X": ["年龄", "收入层次"],
    #         "X": ["教育"],
    #         # "X": ["教育", "工龄"],
    #         "Y": ["违约", "地址"]
    #         # "Y": ["违约"],
    #         # "Y": ["是否单身"]
    #         # "Y": ["是否单身", "信用等级"]
    #     },  # list,自变量和各因变量交叉表
    #     "statistic": True
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/dataAnalysis', json=kwargs, timeout=500)

    """
    =====================================================================
                                    待完成
    =====================================================================
    """

    # ======================= 支持向量机 =============================
    # ======================= 层次聚类 ==============================
    print("total time:{}".format(time.time() - start_time))
    print(res.text)
