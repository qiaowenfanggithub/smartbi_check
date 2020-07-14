# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : data_preprocess

Description :
    1，特征编码
        1）无关的离散变量用onehot
        2）有序的离散变量用序列编码
    2，归一化
        1）归一化
        2）标准化
    3，计算向量相似度

Author : leiliang

Date : 2020/6/24 9:15 上午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import request, Flask, jsonify
import logging

log = logging.getLogger(__name__)
app = Flask(__name__)


# 特征编码
@app.route('/data_preprocess/features_encode', methods=['POST'])
def data_encoder(data: pd.DataFrame, column_name, use_onehot=False, default_value=0):
    log_file = "data_preprocess.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('data_preprocess_features_encode_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        csv_file = request_data["csv_file"]
        column_name = request_data["column_name"]
        use_onehot = request_data["use_onehot"]
        default_value = request_data["default_value"]
    except Excepiton as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_csv(csv_file)
        if use_onehot:
            if column_name not in data.columns:
                raise ValueError("{} not in {} columns".format(column_name, data))
            data = data.join(pd.get_dummies(data[column_name]))
            data.drop([column_name], axis=1, inplace=True)
        else:
            # Replace missing values with "U0"
            data[column_name][data[column_name].isnull()] = default_value
            # convert the distinct cabin letters with incremental integer values
            data[column_name] = pd.factorize(data[column_name].iloc[:, 0])[0]
        # todo:返回的应该是jsonfiy的结果
        return data
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 归一化
@app.route('/data_preprocess/standard', methods=['POST'])
def data_standard(data: pd.DataFrame, column_name, method="normal"):
    log_file = "data_preprocess.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('data_preprocess_standard_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        csv_file = request_data["csv_file"]
        column_name = request_data["column_name"]
    except Excepiton as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_csv(csv_file)
        if method == "normal":
            data[column_name] = (data[column_name] - data[column_name].min()) / (
                    data[column_name].max() - data[column_name].min())
        else:
            data[column_name] = (data[column_name] - data[column_name].mean()) / (data[column_name].std())
        # todo:返回的应该是jsonfiy的结果
        return data
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算向量相似度
@app.route('/data_preprocess/compute_similarity', methods=['POST'])
def data_similarity(data0, data1):
    assert len(data0) == len(data1)
    return np.round(cosine_similarity([data0, data1])[0][1], 8)


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([5, 6, 7])
    # c = np.array([5, 6, 7])
    pd1 = pd.DataFrame(a)
    pd2 = pd.DataFrame(b)
    data = pd.read_csv("PimaIndiansdiabetes.csv")
    # data = pd.read_excel("buy-computer.xlsx")
    print(data)
    encoder_data = data_encoder(data, ["Glucose"])
    print(encoder_data)
    standard_data = data_standard(data, ["Glucose"])
    print(standard_data)
