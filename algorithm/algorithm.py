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
    if algorithm == "linerRegression":
        try:
            from algorithm_liner_regression import linerRegression
        except NotImplementedError as e:
            raise e
        liner_regression = linerRegression(method)
        response_data = exec(method, liner_regression)
        return jsonify(response_data)
    elif algorithm == "polyRegression":
        pass
    elif algorithm == "svm":
        pass
    elif algorithm == "decisionTree":
        pass
    elif algorithm == "randomForest":
        pass
    elif algorithm == "logisticRegression":
        try:
            from algorithm_logistic import logisticAlgorithm
        except NotImplementedError as e:
            raise e
        logistic = logisticAlgorithm(method)
        response_data = exec(method, logistic)
        return jsonify(response_data)
    elif algorithm == "kMeans":
        pass
    elif algorithm == "hierarchicalCluster":
        pass
    else:
        raise ValueError("输入算法参数错误:{}".format(algorithm))


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, port=5000)
