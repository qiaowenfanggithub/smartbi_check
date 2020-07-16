# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : main

Description : 

Author : leiliang

Date : 2020/7/9 4:03 下午

--------------------------------------------------------

"""
from __future__ import print_function
from flask import Flask, request, jsonify
from flask.json import JSONEncoder as _JSONEncoder
import logging
import numpy as np
import pandas as pd
import time
from flask_cors import *
from utils import get_dataframe_from_mysql
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)


class JSONEncoder(_JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)


def init_route():
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    logging.getLogger("requests").setLevel(logging.WARNING)
    try:
        request_data = request.json
        log.info(request_data)
    except Exception as e:
        log.info(e)
        raise e
    log.info("receive request :{}".format(request_data))
    return request_data


def show_results(x, y, model):
    """
    预测结果输出
    :param x: 测试集特征
    :param y: 测试集标签
    :param model: 训练好的模型
    :return: 结果
    """
    # 输出结果展示
    y_predict = model.predict(x)
    y_predict_proba = model.predict_proba(x)
    # accuracy_score = metrics.accuracy_score(y, y_predict)
    # precision_score = metrics.precision_score(y, y_predict)
    # recall_score = metrics.recall_score(y, y_predict)
    # f1_score = metrics.f1_score(y, y_predict)
    confusion_matrix = metrics.confusion_matrix(y, y_predict)
    report = metrics.classification_report(y, y_predict, target_names=model.classes_.tolist())
    roc_auc_res = {}
    for idx, label in enumerate(model.classes_):
        # fpr,tpr用于绘制曲线
        fpr, tpr, threshold = metrics.roc_curve(y, y_predict_proba[:, idx], pos_label=label)
        auc_score = metrics.auc(fpr, tpr)
        roc_auc_res.update({label: {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "threshold": threshold.tolist(),
            "auc_score": auc_score,
        }})


    return {
        "confusion_matrix": confusion_matrix.tolist(),
        "roc_auc_res": roc_auc_res,
        "report": report,
    }


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


# ================================ 决策树-训练 ==============================
@app.route('/algorithm/decisionTree/train', methods=['POST', 'GET'])
def decision_tree_train():
    """
    接口请求参数:{
        "isTrain": "" # True,进行训练还是测试
        "tableName": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "rate": "0.3", # str,测试集训练集分割比例
        "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
        "cv": "10", # str,几折交叉验证
        "param":{
            "criterion": "gini", # 不纯度指标gini、entropy
            "max_depth": "2", # 指定树的最大深度, default=2
            "min_sample_split": "2", # :int, float, optional (default=2)。表示分裂一个内部节点需要的最少样本数。
            "min_samples_leaf": "1", # int, float, optional (default=1)。指定每个叶子节点需要的最少样本数。
        }
    }
    :return:
    """
    log.info('decision_tree_train_init...')
    request_data = init_route()
    try:
        is_train = request_data['isTrain']
        table_name = request_data['tableName']
        X = request_data['X']
        Y = request_data['Y']
        param = request_data['param']
        random_state = int(request_data.get('randomState', 0))
        rate = float(request_data.get('rate', 0))
        cv = int(request_data.get('cv', 0))
    except Exception as e:
        log.info(e)
        raise e
    param["criterion"] = param.get("criterion", ["gini"])
    param["max_depth"] = param.get("max_depth")
    param["max_depth"] = [int(m) for m in param["max_depth"]] if param["max_depth"] else [2]
    param["min_sample_split"] = param.get("min_sample_split")
    param["min_sample_split"] = [int(m) for m in param["min_sample_split"]] if param["min_sample_split"] else [2]
    param["min_samples_leaf"] = param.get("min_samples_leaf")
    param["min_samples_leaf"] = [int(m) for m in param["min_samples_leaf"]] if param["min_samples_leaf"] else [2]
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            raise ValueError("input Y must not bu empty")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        # 测试模型
        if not is_train:
            model_name_list = os.listdir("./model/DecisionTreeClassifier")
            model_name_list.sort()
            lastest_model_path = os.path.join("./model/DecisionTreeClassifier", model_name_list[-1])
            test_model = joblib.load(lastest_model_path)
            x_test = data.loc[:, X].values
            y_test = data[Y[0]].values

            label_list = np.unique(y_test)
            # 输出测试集结果
            test_results = show_results(x_test, y_test, test_model, label_list)

            # 可视化决策树图
            # generate_tree_graph(test_model, x_test.columns, label_list)

            response_data = {"res": test_results,
                             "code": "200",
                             "msg": "ok!"}
            return jsonify(response_data)
        # 训练模式
        else:
            data_x = data.loc[:, X].values
            data_y = data[Y[0]].values
            # 数据分割
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                                random_state=random_state,
                                                                test_size=rate)

            # 模型训练和网格搜索
            clf = DecisionTreeClassifier(random_state=random_state)
            model = GridSearchCV(clf, param, cv=cv)
            model.fit(x_train, y_train)
            best_param = model.best_params_
            model = DecisionTreeClassifier(**best_param, random_state=random_state).fit(x_test, y_test)

            if not os.path.exists("./model/DecisionTreeClassifier/"):
                os.mkdir("./model/DecisionTreeClassifier/")
            joblib.dump(model, "./model/DecisionTreeClassifier/{}.pkl".format(
                time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())))

            label_list = np.unique(y_train)
            # 输出测试集结果
            test_results = show_results(x_test, y_test, model, label_list)

            # 可视化决策树图
            # generate_tree_graph(model, X, label_list)

            response_data = {"res": test_results,
                             "graph": "",
                             "code": "200",
                             "msg": "ok!"}
            return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 决策树-预测 ==============================
@app.route('/algorithm/decisionTree/predict', methods=['POST', 'GET'])
def decision_tree_predict():
    """
    接口请求参数:{
        "oneSample": False, # 是否批量上传数据进行预测
        "tableName": "", # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    }
    :return:
    """
    log.info('decision_tree_predict_init...')
    request_data = init_route()
    try:
        one_sample = request_data['oneSample']
        table_name = request_data['tableName']
        X = request_data['X']
    except Exception as e:
        log.info(e)
        raise e
    try:
        model_name_list = os.listdir("./model/DecisionTreeClassifier")
        model_name_list.sort()
        lastest_model_path = os.path.join("./model/DecisionTreeClassifier", model_name_list[-1])
        model = joblib.load(lastest_model_path)
        if one_sample:
            res = model.predict([X])
        else:
            # 从数据库拿数据
            try:
                sql_sentence = "select {} from {};".format(",".join(X), table_name)
                data = get_dataframe_from_mysql(sql_sentence)
            except Exception as e:
                log.info(e.args)
                raise e
            log.info("输入数据大小:{}".format(len(data)))
            res = model.predict(data.values)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})
    response_data = {"res": res,
                     "code": "200",
                     "msg": "ok!"}
    return jsonify(response_data)


# ================================ 逻辑回归-训练 ==============================
@app.route('/algorithm/logistics/train', methods=['POST', 'GET'])
def logistics_train():
    """
    接口请求参数:{
        "isTrain": "", # True,进行训练还是测试
        "tableName": "", # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "rate": "0.3", # str,测试集训练集分割比例
        "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
        "cv": "10", # str,几折交叉验证
        "param":{
            "penalty": "l2", # str,惩罚项
            "C": "2", # str,惩罚项系数
            "solver": "saga", # str，优化算法
            "max_ter": "1000", # str，最大迭代步数
        }
    :return:
    """
    log.info('logistics_train_init...')
    request_data = init_route()
    try:
        is_train = request_data['isTrain']
        table_name = request_data['tableName']
        X = request_data['X']
        Y = request_data['Y']
        param = request_data['param']
        random_state = int(request_data.get('randomState', 0))
        rate = float(request_data.get('rate', 0))
        cv = int(request_data.get('cv', 0))
    except Exception as e:
        log.info(e)
        raise e
    param["penalty"] = [param.get("penalty", "")]
    param["C"] = [float(param.get("C", 0))]
    # 默然saga随机梯度下降
    param["solver"] = [param.get("solver", "saga")]
    param["max_iter"] = [int(param.get("max_iter", 1000))]
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            raise ValueError("input Y must not bu empty")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    # 数据类型统一为float
    data[X] = data[X].astype("float16")
    data[Y] = data[Y].astype("str")
    try:
        # 测试模型
        if not is_train:
            model_name_list = os.listdir("./model/LogisticRegression")
            model_name_list.sort()
            lastest_model_path = os.path.join("./model/LogisticRegression", model_name_list[-1])
            test_model = joblib.load(lastest_model_path)
            x_test = data.loc[:, X].values
            y_test = data[Y[0]].values
            # 输出测试集分类结果
            classifier_res = show_results(x_test, y_test, test_model)

            # 拟合优度结果
            try:
                import statsmodels.api as sm
            except:
                raise ImportError("statsmodels.api cannot import")
            try:
                x = sm.add_constant(x_test)
                logit_stats_model = sm.Logit(y_test.astype("int16"), x)
                logit_stats_res = logit_stats_model.fit()
                # 拟合优度
                logit_regression_res = logit_stats_res.summary().tables[0].data
                # 系数解读
                coef_explain = logit_stats_res.summary().tables[1].data
            except Exception as e:
                log.error("statsmodels analysis error")
                raise e

            response_data = {"classifier_res": classifier_res,
                             "logit_regression_res": logit_regression_res,
                             "coef_explain": coef_explain,
                             "code": "200",
                             "msg": "ok!"}
            return jsonify(response_data)
        # 训练模式
        else:
            data_x = data.loc[:, X].values
            data_y = data[Y[0]].values
            # 数据分割
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                                random_state=random_state,
                                                                test_size=rate)

            # 模型训练和网格搜索
            clf = LogisticRegression(random_state=random_state)
            model = GridSearchCV(clf, param, cv=cv)
            model.fit(x_train, y_train)
            best_param = model.best_params_
            model = LogisticRegression(**best_param, random_state=random_state).fit(x_test, y_test)

            if not os.path.exists("./model/LogisticRegression/"):
                os.mkdir("./model/LogisticRegression/")
            joblib.dump(model, "./model/LogisticRegression/{}.pkl".format(
                time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())))

            # 输出测试集结果
            classifier_res = show_results(x_test, y_test, model)

            # 拟合优度结果
            try:
                import statsmodels.api as sm
            except:
                raise ImportError("statsmodels.api cannot import")
            try:
                x = sm.add_constant(x_test)
                logit_stats_model = sm.Logit(y_test.astype("int16"), x)
                logit_stats_res = logit_stats_model.fit()
                # 拟合优度
                logit_regression_res = logit_stats_res.summary().tables[0].data
                # 系数解读
                coef_explain = logit_stats_res.summary().tables[1].data
            except Exception as e:
                log.error("statsmodels analysis error")
                raise e

            response_data = {"classifier_res": classifier_res,
                             "logit_regression_res": logit_regression_res,
                             "coef_explain": coef_explain,
                             "code": "200",
                             "msg": "ok!",
                             }
            return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 逻辑回归-预测 ==============================
@app.route('/algorithm/logistics/predict', methods=['POST', 'GET'])
def logistics_predict():
    """
    接口请求参数:{
        "oneSample": False,  # 是否批量上传数据进行预测
        "tableName": "buy_computer_new",  # str,数据库表名
        "X": ["x1", "x2"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        }
    :return:
    """
    log.info('logistics_predict_init...')
    request_data = init_route()
    try:
        one_sample = request_data['oneSample']
        table_name = request_data['tableName']
        X = request_data['X']
    except Exception as e:
        log.info(e)
        raise e
    try:
        model_name_list = os.listdir("./model/LogisticRegression")
        model_name_list.sort()
        lastest_model_path = os.path.join("./model/LogisticRegression", model_name_list[-1])
        model = joblib.load(lastest_model_path)
        if one_sample:
            res = model.predict([X])
        else:
            # 从数据库拿数据
            try:
                sql_sentence = "select {} from {};".format(",".join(X), table_name)
                data = get_dataframe_from_mysql(sql_sentence)
            except Exception as e:
                log.info(e.args)
                raise e
            log.info("输入数据大小:{}".format(len(data)))
            res = model.predict(data.values)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})
    response_data = {"res": res,
                     "code": "200",
                     "msg": "ok!"}
    return jsonify(response_data)


# ================================ K-means 训练 ==============================
@app.route('/algorithm/kMeans', methods=['POST', 'GET'])
def kmeans():
    """
    接口请求参数:{
        "tableName": "", # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
        "param":{
            "n_clusters": "2", # str,聚类中心数量，默认2个
            "max_ter": "1000", # str，最大迭代步数，默认1000个
        }
    :return:
    """
    log.info('logistics_train_init...')
    request_data = init_route()
    try:
        table_name = request_data['tableName']
        X = request_data['X']
        param = request_data['param']
        random_state = int(request_data.get('randomState', 0))
    except Exception as e:
        log.info(e)
        raise e
    param["n_clusters"] = param["n_clusters"]
    param["max_iter"] = int(param.get("max_iter", 1000))
    # 从数据库拿数据
    try:
        sql_sentence = "select {} from {};".format(",".join(X), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    data_x = data.loc[:, X].values
    try:
        res = []
        for n_cluster in param["n_clusters"]:
            model = KMeans(n_clusters=int(n_cluster),
                           max_iter=param["max_iter"],
                           random_state=random_state)
            # todo:是否需要可视化，多维需要转成二维展示
            pred = model.fit(data_x)
            # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            # plt.show()
            sse = model.inertia_
            sh = metrics.silhouette_score(data_x, model.labels_, metric="euclidean")
            ch = metrics.calinski_harabaz_score(data_x, model.labels_)
            res.append({
                "pred": model.labels_,
                "sse": sse,  # 误差平方和
                "sh": sh,  # 轮廓系数
                "ch": ch,  # Calinski-Harabasz分数值
            })

        response_data = {"res": res,
                         "n_clusters_list": param["n_clusters"],
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 随机森林-训练 ==============================
@app.route('/algorithm/randomForest/train', methods=['POST', 'GET'])
def random_forest_train():
    """
    接口请求参数:{
        "isTrain": "", # True,进行训练还是测试
        "tableName": "", # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "rate": "0.3", # str,测试集训练集分割比例
        "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
        "cv": "3", # str,几折交叉验证
        "param":{
            "n_estimators": [100, 200], # list,树的个数
            "criterion": ["gini", "entropy"], # list,树划分准则
            "max_features": [""], # list，用于训练的最大特征数量
            "max_depth": [5, 10, 15], # list，树的最大深度列表
            "min_samples_split": [2, 3, 4], # list， 内部节点再划分所需最小样本数
            "min_samples_leaf": [2, 3, 4], # list，叶子节点最少样本数
            }
        }
    :return:
    """
    log.info('random_forest_train_init...')
    request_data = init_route()
    try:
        is_train = request_data['isTrain']
        table_name = request_data['tableName']
        X = request_data['X']
        Y = request_data['Y']
        param = request_data['param']
        random_state = int(request_data.get('randomState', 0))
        rate = float(request_data.get('rate', 0))
        cv = int(request_data.get('cv', 0))
    except Exception as e:
        log.info(e)
        raise e
    param["n_estimators"] = param["n_estimators"]
    param["criterion"] = param["criterion"]
    param["max_features"] = param["max_features"]
    param["max_depth"] = param["max_depth"]
    param["min_samples_split"] = param["min_samples_split"]
    param["min_samples_leaf"] = param["min_samples_leaf"]
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            raise ValueError("input Y must not bu empty")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        # 测试模型
        if not is_train:
            model_name_list = os.listdir("./model/RandomForestClassifier")
            model_name_list.sort()
            lastest_model_path = os.path.join("./model/RandomForestClassifier", model_name_list[-1])
            test_model = joblib.load(lastest_model_path)
            x_test = data.loc[:, X].values
            y_test = data[Y[0]].values

            label_list = np.unique(y_test)
            # 输出测试集结果
            test_results = show_results(x_test, y_test, test_model, label_list)

            # 可视化决策树图
            # generate_tree_graph(test_model, x_test.columns, label_list)

            response_data = {"res": test_results,
                             "code": "200",
                             "msg": "ok!"}
            return jsonify(response_data)
        # 训练模式
        else:
            data_x = data.loc[:, X].values
            data_y = data[Y[0]].values
            # 数据分割
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                                random_state=random_state,
                                                                test_size=rate)

            # 模型训练和网格搜索
            clf = RandomForestClassifier(random_state=random_state)
            model = GridSearchCV(clf, param, cv=cv, scoring="roc_auc")
            model.fit(x_train, y_train)
            best_param = model.best_params_
            model = RandomForestClassifier(**best_param, random_state=random_state).fit(x_test, y_test)

            if not os.path.exists("./model/RandomForestClassifier/"):
                os.mkdir("./model/RandomForestClassifier/")
            joblib.dump(model, "./model/RandomForestClassifier/{}.pkl".format(
                time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())))

            label_list = np.unique(y_train)
            # 输出测试集结果
            test_results = show_results(x_test, y_test, model, label_list)

            # 可视化决策树图
            # generate_tree_graph(model, X, label_list)

            response_data = {"res": test_results,
                             "code": "200",
                             "msg": "ok!"}
            return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 随机森林-预测 ==============================
@app.route('/algorithm/randomForest/predict', methods=['POST', 'GET'])
def random_forest_predict():
    """
    接口请求参数:{
        "oneSample": False, # 是否批量上传数据进行预测
        "tableName": "", # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    }
    :return:
    """
    log.info('random_forest_predict_init...')
    request_data = init_route()
    try:
        one_sample = request_data['oneSample']
        table_name = request_data['tableName']
        X = request_data['X']
    except Exception as e:
        log.info(e)
        raise e
    try:
        model_name_list = os.listdir("./model/RandomForestClassifier")
        model_name_list.sort()
        lastest_model_path = os.path.join("./model/RandomForestClassifier", model_name_list[-1])
        model = joblib.load(lastest_model_path)
        if one_sample:
            res = model.predict([X])
        else:
            # 从数据库拿数据
            try:
                sql_sentence = "select {} from {};".format(",".join(X), table_name)
                data = get_dataframe_from_mysql(sql_sentence)
            except Exception as e:
                log.info(e.args)
                raise e
            log.info("输入数据大小:{}".format(len(data)))
            res = model.predict(data.values)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})
    response_data = {"res": res,
                     "code": "200",
                     "msg": "ok!"}
    return jsonify(response_data)


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, port=5000)
