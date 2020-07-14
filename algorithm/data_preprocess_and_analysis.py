# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : data_preprocess_and_analysis

Description : 

Author : leiliang

Date : 2020/6/28 4:07 下午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from flask import request, Flask, jsonify

log = logging.getLogger(__name__)
app = Flask(__name__)


# ======================= 算法预处理 =============================
# 特征编码
@app.route('/features_encode', methods=['POST'])
def data_encoder():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('features_encode_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
        use_onehot = request_data.get("use_onehot")
        default_value = request_data.get("default_value")
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        if isinstance(column_name, list):
            for col in column_name:
                if use_onehot:
                    if col not in data.columns:
                        raise ValueError("{} not in {} columns".format(col, data))
                    onehot_data = pd.get_dummies(data[col])
                    onehot_data_col = onehot_data.columns
                    onehot_data.rename(columns={onehot_data_col[0]: col + "_" + onehot_data_col[0],
                                        onehot_data_col[1]: col + "_" + onehot_data_col[1]}, inplace=True)
                    data = data.join(onehot_data)
                    data.drop([col], axis=1, inplace=True)
                else:
                    # Replace missing values with "0"
                    data[col][data[col].isnull()] = default_value
                    # convert the distinct cabin letters with incremental integer values
                    data[col] = pd.factorize(data[col].iloc[:, 0])[0]

        else:
            if use_onehot:
                if column_name not in data.columns:
                    raise ValueError("{} not in {} columns".format(column_name, data))
                onehot_data = pd.get_dummies(data[column_name])
                onehot_data_col = onehot_data.columns
                onehot_data.rename(columns={onehot_data_col[0]: column_name + "_" + onehot_data_col[0],
                                    onehot_data_col[1]: column_name + "_" + onehot_data_col[1]}, inplace=True)
                data = data.join(onehot_data)
                data.drop([column_name], axis=1, inplace=True)
            else:
                # Replace missing values with "0"
                data[column_name][data[column_name].isnull()] = default_value
                # convert the distinct cabin letters with incremental integer values
                data[column_name] = pd.factorize(data[column_name].iloc[:, 0])[0]
        return jsonify({"data": data.to_json(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 归一化
@app.route('/features_standard', methods=['POST'])
def data_standard():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('features_standard_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
        method = request_data["method"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        if isinstance(column_name, list):
            for col in column_name:
                if method == "normal":
                    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                else:
                    data[col] = (data[col] - data[col].mean()) / (data[col].std())
        else:
            if method == "normal":
                data[column_name] = (data[column_name] - data[column_name].min()) / (
                        data[column_name].max() - data[column_name].min())
            else:
                data[column_name] = (data[column_name] - data[column_name].mean()) / (data[column_name].std())
        return jsonify({"data": data.to_json(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算向量相似度
@app.route('/similarity', methods=['POST'])
def data_similarity():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('similarity_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name_x = request_data["column_name_x"]
        column_name_y = request_data["column_name_y"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        data0 = data[column_name_x]
        data1 = data[column_name_y]
        res = np.round(cosine_similarity([data0, data1])[0][1], 8)
        response_data = {"data": res, "code": "200", "msg": "ok"}
        return jsonify(response_data)
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# ======================= 一般描述统计 =============================
# 计算平均值
@app.route('/mean', methods=['POST'])
def data_mean():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('mean_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify({"data": data[column_name].mean(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算中位数
@app.route('/median', methods=['POST'])
def data_median():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('median_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify({"data": data[column_name].median(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算众数
@app.route('/mode', methods=['POST'])
def data_mode():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('mode_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify({"data": data[column_name].mode(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算方差
@app.route('/var', methods=['POST'])
def data_var():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('var_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify({"data": data[column_name].var(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算标准差
@app.route('/std', methods=['POST'])
def data_std():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('std_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify({"data": data[column_name].std(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 计算极数
@app.route('/extreme', methods=['POST'])
def data_extreme():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('extreme_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify(
            {"data": {"max": data[column_name].max(), "min": data[column_name].min()}, "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# ======================= 数据概况 =============================
# 变异系数
@app.route('/data_cv', methods=['POST'])
def data_cv():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('data_cv_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify(
            {"data": data[column_name].mean() / data[column_name].std(), "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 频数分布
@app.route('/data_count', methods=['POST'])
def data_count():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('data_count_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify(
            {"data": {"index": data[column_name].index.values.tolist(),
                      "count": data[column_name].value_counts().values.tolist()},
             "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 四分位数
@app.route('/quantity', methods=['POST'])
def data_quantity():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('quantity_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        return jsonify(
            {"data": {"index": data[column_name].index.values.tolist(),
                      "values": data[column_name].quantile().values.tolist()},
             "code": "200", "msg": "ok"})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# ======================= 探索性分析 =============================
# 频数统计（0-1直方图）
@app.route('/plot_count', methods=['POST'])
def data_count_plot():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('plot_count_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        column_name = request_data["column_name"]
        hue = request_data["hue"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        sns.countplot(x=column_name, hue=hue, data=data, palette="Pastel2")
        plt.xlabel(column_name)
        plt.title("{} by {}".format(hue, column_name))
        save_path = "count_plot_{}_by_{}.png".format(hue, column_name)
        plt.savefig(save_path)
        return jsonify({"data": "", "code": "200", "msg": "plot count success:{}".format(save_path)})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 相关系数矩阵
@app.route('/plot_corr', methods=['POST'])
def data_corr_plot():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('plot_corr_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        corr = data.corr()
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                    linewidths=0.2, cmap="YlGnBu", annot=True)
        plt.title("Correlation between variables")
        save_path = "corr_plot.png"
        plt.savefig(save_path)
        return jsonify({"data": "", "code": "200", "msg": "plot corr success:{}".format(save_path)})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


# 交叉散点图（每个X与Y的散点图）
@app.route('/plot_scatter', methods=['POST'])
def data_scatter_plot():
    log_file = "data_preprocess_and_analysis.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    log.info('plot_scatter_init...')
    try:
        request_data = request.json
        log.info("receive request :{}".format(request_data))
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "receive request data error:{}".format(e)})
    try:
        json_data = request_data["json_data"]
        col_name_X = request_data["col_name_x"]
        col_name_Y = request_data["col_name_y"]
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "get input param error:{}".format(e)})
    try:
        data = pd.read_json(json_data)
        sns.scatterplot(data[col_name_X], data[col_name_Y])
        plt.title("{} by {}".format(col_name_Y, col_name_X))
        save_path = "scatter_plot_{}_by_{}.png".format(col_name_Y, col_name_X)
        plt.savefig(save_path)
        return jsonify({"data": "", "code": "200", "msg": "plot scatter success:{}".format(save_path)})
    except Exception as e:
        log.exception(e.args)
        response_data = {"data": "error", "code": "500", "msg": "run error:{}".format(e.args)}
        return jsonify(response_data)


if __name__ == '__main__':
    app.run("0.0.0.0", 5000, debug=False)
