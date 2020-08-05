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
from evaluate import evaluateModel

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
        liner_regression = linerRegression(method)
        response_data = exec(method, liner_regression)
        return jsonify(response_data)
    # 多项式回归（qwf）--》训练、预测
    elif algorithm == "polyRegression":
        try:
            from algorithm_poly_regression import polyRegression
        except NotImplementedError as e:
            raise e
        poly_regression = polyRegression(method)
        response_data = exec(method, poly_regression)
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
        random_forest = decisionTree(method)
        response_data = exec(method, random_forest)
        return jsonify(response_data)
    # 随机森林（lei）
    elif algorithm == "randomForest":
        try:
            from algorithm_random_forest import randomForest
        except NotImplementedError as e:
            raise e
        random_forest = randomForest(method)
        response_data = exec(method, random_forest)
        return jsonify(response_data)
    # 逻辑回归（lei）--》训练、评估、预测
    elif algorithm == "logisticRegression":
        try:
            from algorithm_logistic import logistic
        except NotImplementedError as e:
            raise e
        logistics = logistic(method)
        response_data = exec(method, logistics)
        return jsonify(response_data)
    # k-means聚类（lei）--》训练、预测
    elif algorithm == "kMeans":
        try:
            from algorithm_kmeans import kMeans
        except NotImplementedError as e:
            raise e
        logistics = kMeans(method)
        response_data = exec(method, logistics)
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


# ================================ 评估总入口 ==============================
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


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, port=5000)
