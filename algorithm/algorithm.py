# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm

Description : 数据分析平台-算法-主函数

Author : leiliang

Date : 2020/7/27 9:12 上午

--------------------------------------------------------

"""
from __future__ import print_function
from flask import Flask, jsonify
from flask.json import JSONEncoder as _JSONEncoder
from flask_cors import *
from utils import *
from flask import request
from base64_to_png import base64_to_img

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


app.json_encoder = JSONEncoder
app.config['JSON_AS_ASCII'] = False


def exec(method, algorithm):
    if method == 'train':
        log.info(str(algorithm) + '_train_init...')
        response_data = algorithm.train()
    elif method == "evaluate":
        log.info(str(algorithm) + '_evaluate_init...')
        response_data = algorithm.evaluate()
    elif method == "predict":
        log.info(str(algorithm) + '_predict_init...')
        response_data = algorithm.predict()
    elif method == "visualization":
        log.info(str(algorithm) + '_visualization_init...')
        response_data = algorithm.visualization()
    else:
        raise ValueError("input method:{} error".format(method))
    return response_data


# ================================ 算法总入口 ==============================
@app.route('/algorithm/<algorithm>/<method>', methods=['POST', 'GET'])
def main(algorithm, method):
    """
    动态路由输入算法和方法，动态解析并返回结果
    :param algorithm: ["linerRegression", "polyRegression", "svm", "decisionTree", "randomForest", "logistic", "kMeans", "hierarchicalCluster"]
    :param method: ["train", "evaluate", "predict"]
    :return:train和evaluate模式下返回分类、回归或聚类的结果，predict下返回预测的结果
    """
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    logging.getLogger().setLevel(logging.WARNING)
    # 线性回归（lei）--》训练、预测
    if algorithm == "linerRegression":
        try:
            from algorithm_liner_regression import linerRegression
        except NotImplementedError as e:
            raise e
        liner_regression_alg = linerRegression(method)
        response_data = exec(method, liner_regression_alg)

        # check base64 png
        for data in response_data["res"]:
            if "base64" in data:
                base64_to_img(data["base64"])
        return jsonify(response_data)
    # 多项式回归（qwf）--》训练、预测
    elif algorithm == "polyLinerRegression":
        try:
            from algorithm_poly_regression import polyRegression
        except NotImplementedError as e:
            raise e
        poly_regression_alg = polyRegression(method)
        response_data = exec(method, poly_regression_alg)
        return jsonify(response_data)
    # 支持向量机（hyj）--》训练、评估、预测
    elif algorithm == "svmClassifier":
        pass
    # 决策树（qwf）--》训练、评估、预测
    elif algorithm == "decisionTree":
        try:
            from algorithm_decision_tree import decisionTree
        except NotImplementedError as e:
            raise e
        decision_tree_alg = decisionTree(method)
        response_data = exec(method, decision_tree_alg)
        return jsonify(response_data)
    # 随机森林（lei）
    elif algorithm == "randomForest":
        try:
            from algorithm_random_forest import randomForest
        except NotImplementedError as e:
            raise e
        random_forest_alg = randomForest(method)
        response_data = exec(method, random_forest_alg)
        return jsonify(response_data)
    # 逻辑回归（lei）--》训练、评估、预测
    elif algorithm == "logisticRegression":
        try:
            from algorithm_logistic import logistic
        except NotImplementedError as e:
            raise e
        logistics_alg = logistic(method)
        response_data = exec(method, logistics_alg)
        return jsonify(response_data)
    # k-means聚类（lei）--》训练、预测
    elif algorithm == "kMeans":
        try:
            from algorithm_kmeans import kMeans
        except NotImplementedError as e:
            raise e
        kmeans_alg = kMeans(method)
        response_data = exec(method, kmeans_alg)
        return jsonify(response_data)
    # 层次聚类（hyj）--》训练、预测
    elif algorithm == "hierarchicalCluster":
        pass
    else:
        raise ValueError("输入算法参数错误:{}".format(algorithm))


# ================================ 算法模型查询接口 ==============================
@app.route('/algorithm/selectModel/<algorithm>', methods=['POST', 'GET'])
def select_model(algorithm):
    try:
        # todo:需要筛选出同一用户的编号下的
        sql = "SELECT name FROM algorithm_model WHERE type='{}';".format(algorithm)
        res_tuple = exec_select_sql(sql)
        res = [d[0] for d in res_tuple]
        response_data = {
            "model_name_list": res,
            "code": "200",
            "msg": "ok!",
        }
        return jsonify(response_data)
    except Exception as e:
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


# ================================ 评估总入口 ==============================
@app.route('/algorithm/evaluate', methods=['POST', 'GET'])
def evaluate():
    """
    前端传过来的参数
    {
        "algorithm": "",
        "model": "",
        "table": "",
        "X": "",
        "Y": "",
        "show_options": ["report", "matrix", "roc"]
    }
    :return:
    """
    try:
        from evaluate import evaluateModel
    except NotImplementedError as e:
        raise e
    response_data = evaluateModel().model_evaluate()
    return jsonify(response_data)


# ================================ 预测总入口 ==============================
@app.route('/algorithm/predict', methods=['POST', 'GET'])
def predict():
    """
    前端传过来的参数
    {
        "algorithm": "",
        "model": "",
        "oneSample": False
        "table": "",
        "X": "",
    }
    :return:
    """
    try:
        from predict import predictModel
    except NotImplementedError as e:
        raise e
    response_data = predictModel().model_predict()
    return jsonify(response_data)


# ================================ 模型查看特征接口 ==============================
@app.route('/algorithm/checkModelFeatures', methods=['POST', 'GET'])
def check_model_features():
    request_data = request.json
    try:
        algorithm = request_data["algorithm"]
        model = request_data["model"]
        sql = "SELECT characteristic_column FROM algorithm_model WHERE type='{}' and name='{}';".format(algorithm,
                                                                                                        model)
        res_tuple = exec_select_sql(sql)
        res = res_tuple[0][0]
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


# ================================ 数据预处理-编码和归一化 ==============================
@app.route('/algorithm/dataProcess/<method>', methods=['POST', 'GET'])
def data_preprocess(method):
    """
    数据预处理请求参数{
        "tableName": "",  # str,数据库表名
        "encoder":{
            "oneHot": [],  # list,需要使用onehot编码的特征列
            "factorize": [] # list,需要使用序列编码的特征列
        },
        "normalize":{
            "normal": [],  # list,需要使用normal标准化的特征列
            "standard": [] # list,需要使用归一化的特征列
        }
    }
    :param method:
    :return:
    """
    request_data = request.json
    try:
        table_name = request_data["tableName"]
        # 获取数据从数据表
        sql = "select * from {};".format("`" + table_name + "`")
        table_data = get_dataframe_from_mysql(sql, database='sophia_data')
        encoder_config = request_data.get("encoder")
        normalize_config = request_data.get("normalize")
        if method not in ["encoder", "normalize"]:
            raise ValueError("input dataProcess method:{} is not support".format(method))
        if method == "encoder":
            if not any([i in ["oneHot", "factorize"] for i in encoder_config]):
                raise ValueError("input encoder config:{} is not correct".format(encoder_config))
            if encoder_config.get("oneHot") and encoder_config["oneHot"][0] != "":
                table_data = data_encoder(table_data, encoder_config.get("oneHot"), use_onehot=True)
            if encoder_config.get("factorize") and encoder_config["factorize"][0] != "":
                table_data = data_encoder(table_data, encoder_config.get("factorize"))
        if method == "normalize":
            if not any([i in ["minMaxScale", "standard"] for i in normalize_config]):
                raise ValueError("input encoder config:{} is not correct".format(normalize_config))
            if normalize_config.get("minMaxScale") and normalize_config["minMaxScale"][0] != "":
                table_data = data_standard(table_data, normalize_config.get("minMaxScale"), method="minMaxScale")
            if normalize_config.get("standard") and normalize_config["standard"][0] != "":
                table_data = data_standard(table_data, normalize_config.get("standard"), method="standard")
        res = {
            "title": "数据预处理后的数据",
            "row": table_data.index.values.tolist(),
            "col": table_data.columns.values.tolist(),
            "data": table_data.values.tolist()
        }
        response_data = {"res": transform_table_data_to_html(res),
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, port=5000)
