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
        # for data in response_data["res"]:
        #     if "base64" in data:
        #         base64_to_img(data["base64"])
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
        try:
            from algorithm_svm_classifier import svmClassifier
        except NotImplementedError as e:
            raise e
        svm_classifier_alg = svmClassifier(method)
        response_data = exec(method, svm_classifier_alg)
        return jsonify(response_data)
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
    # 多层感知机（lei）
    elif algorithm == "mlpClassifier":
        try:
            from algorithm_mlp import mlpClassifier
        except NotImplementedError as e:
            raise e
        mlp_alg = mlpClassifier(method)
        response_data = exec(method, mlp_alg)
        return jsonify(response_data)
    # 多层感知机（lei）
    elif algorithm == "adaboostClassifier":
        try:
            from algorithm_adaboost import adaboostClassifier
        except NotImplementedError as e:
            raise e
        adaboost_alg = adaboostClassifier(method)
        response_data = exec(method, adaboost_alg)
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
        try:
            from algorithm_hierarchical_cluster import hierarchicalCluster
        except NotImplementedError as e:
            raise e
        hie_cluster = hierarchicalCluster(method)
        response_data = exec(method, hie_cluster)
        return jsonify(response_data)
    else:
        log.exception("Exception Logged")
        raise ValueError("输入算法参数错误:{}".format(algorithm))


# ================================ 算法模型查询接口 ==============================
@app.route('/algorithm/selectModel/<algorithm>', methods=['POST', 'GET'])
def select_model(algorithm):
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
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
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
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
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    try:
        from predict import predictModel
    except NotImplementedError as e:
        raise e
    response_data = predictModel().model_predict()
    return jsonify(response_data)


# ================================ 模型查看特征接口 ==============================
@app.route('/algorithm/checkModelFeatures', methods=['POST', 'GET'])
def check_model_features():
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
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
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    request_data = request.json
    try:
        table_name = request_data["tableId"]
        # 获取数据从数据表
        sql = "select * from {};".format("`" + table_name + "`")
        table_data = get_dataframe_from_mysql(sql, database='sophia_data')
        if method not in ["encoder", "normalize"]:
            raise ValueError("input dataProcess method:{} is not support".format(method))
        if method == "encoder":
            encoder_config = request_data.get("encoder")
            one_hot_config = encoder_config.get("oneHot")
            factorize_config = encoder_config.get("factorize")
            if set(one_hot_config).intersection(set(factorize_config)):
                raise ValueError("数据处理字段重复:{}".format(set(one_hot_config).intersection(set(factorize_config))))
            if one_hot_config and one_hot_config[0] != "":
                table_data = data_encoder(table_data, one_hot_config, use_onehot=True)
            if factorize_config and factorize_config[0] != "":
                table_data = data_encoder(table_data, factorize_config)
        if method == "normalize":
            normalize_config = request_data.get("normalize")
            min_max_scale_config = normalize_config.get("minMaxScale")
            standard_config = normalize_config.get("standard")
            if set(min_max_scale_config).intersection(set(standard_config)):
                raise ValueError("数据处理字段重复:{}".format(set(min_max_scale_config).intersection(set(standard_config))))
            if min_max_scale_config and min_max_scale_config[0] != "":
                table_data = data_standard(table_data, min_max_scale_config, method="minMaxScale")
                table_data = format_dataframe(table_data, {k: ".4f" for k in min_max_scale_config})
            if standard_config and standard_config[0] != "":
                table_data = data_standard(table_data, standard_config, method="standard")
                table_data = format_dataframe(table_data, {k: ".4f" for k in standard_config})
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
        log.exception("Exception Logged")
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


# ================================ 保存模型数据接口 ==============================
@app.route('/algorithm/saveModel', methods=['POST', 'GET'])
def save_model():
    """
    保存模型信息到数据库
    数据预处理请求参数{
        "key": [], # sql数据表的字段名
        "value": [], # sql数据表的value值
        "table": [], # # sql数据表
    }
    :param method:
    :return:
    """
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    logging.getLogger().setLevel(logging.WARNING)
    request_data = request.json
    try:
        table = request_data.get("table", "algorithm_model")
        key = request_data["key"]
        value = request_data["value"]
        # 将模型信息入库
        exec_insert_sql(table, key, value)
        # 将临时文件model_tmp里的模型文件转移到正式文件model
        algorithm_name = value[2]
        model_name = value[1]
        if not os.path.exists("./model/{}".format(algorithm_name)):
            os.makedirs("./model/{}/".format(algorithm_name))
        a = os.system("cp ./model_tmp/{0}/{1}.pkl ./model/{0}".format(algorithm_name, model_name))
        if a != 0:
            raise FileNotFoundError("execute file copy failed")
        response_data = {"res": "",
                         "code": "200",
                         "msg": "exec sql successful"}
        return jsonify(response_data)
    except Exception as e:
        log.exception("Exception Logged")
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


# ================================ 数据探索 ==============================
@app.route('/algorithm/dataAnalysis', methods=['POST', 'GET'])
def data_analysis():
    """
    数据探索接口参数
    {
        "tableName": "" # str,数据库表名
        "dataInfo": [] # 字段列表 list
        "count": [] # 字段列表 list
        "count_hue": "" # 频率图参数 str
        "box": [] # 字段列表 list
        "pie": [] # 字段列表 list
        "pairPlot": [] # 字段列表 list
        "heatMap": [] # 字段列表 list
        "yCorr": [] # 字段列表 list
    }
    :return:
    """
    log_file = "algorithm.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    logging.getLogger().setLevel(logging.WARNING)
    request_data = request.json
    CLASSIFIER = {"svmClassifier", "decisionTree", "randomForest", "logisticRegression"}
    REGRESSION = {"linerRegression", "polyLinerRegression"}
    CLUSTER = {"kMeans", "hierarchicalCluster"}
    res = []
    try:
        table_name = request_data["tableName"]
        count = request_data.get("count", [""])
        count_hue = request_data.get("count_hue")
        dist = request_data.get("dist", [""])
        box = request_data.get("box", [""])
        pie = request_data.get("pie", [""])
        pairPlot = request_data.get("pairPlot", [""])
        heatMap = request_data.get("heatMap", [""])
        yCorr = request_data.get("yCorr")

        # 获取数据从数据表
        sql = "select * from {};".format("`" + table_name + "`")
        table_data = get_dataframe_from_mysql(sql, database='sophia_data')

        # 基本统计信息
        table_data = table_data.astype("float")
        data = table_data.describe()
        data = format_dataframe(data, {k: ".4f" for k in data.columns})
        res.append(transform_table_data_to_html({
            "data": data.values.tolist(),
            "title": "描述性统计分析",
            "col": data.columns.tolist(),
            "row": data.index.tolist()
        }))

        # 频率分布直方图
        if count[0]:
            for x in count:
                if not count_hue:
                    sns.countplot(table_data[x])
                    # 显示纵轴标签
                    plt.ylabel("count")
                    plt.xlabel("{}".format(x))
                    # 显示图标题
                    # plt.title("{} - frequency distribution histogram".format(x))
                    res.append({
                        "title": "{} - 频率分布直方图".format(x),
                        "base64": "{}".format(plot_and_output_base64_png(plt))
                    })
                else:
                    sns.countplot(x=x, hue=count_hue, data=table_data)
                    # 显示纵轴标签
                    plt.ylabel("count")
                    plt.xlabel("{}".format(x))
                    # 显示图标题
                    # plt.title("{} - frequency distribution histogram".format(x))
                    res.append({
                        "title": "{} - 频率分布直方图".format(x),
                        "base64": "{}".format(plot_and_output_base64_png(plt))
                    })
        # 数据分布图
        if dist[0]:
            for x in dist:
                sns.distplot(table_data[x], kde=False)
                # 显示纵轴标签
                plt.xlabel("区间")
                plt.ylabel("{}".format(x))
                # 显示图标题
                # plt.title("{} - frequency distribution histogram".format(x))
                res.append({
                    "title": "{} - 数据分布图".format(x),
                    "base64": "{}".format(plot_and_output_base64_png(plt))
                })
        # 箱型图
        if box[0]:
            for x in box:
                sns.boxplot(table_data[x], palette="Set2", orient="v")
                # 显示纵轴标签
                # plt.ylabel("frequency")
                plt.xlabel("{}".format(x))
                # 显示图标题
                # plt.title("{} - frequency distribution histogram".format(x))
                res.append({
                    "title": "{} - 箱型图".format(x),
                    "base64": "{}".format(plot_and_output_base64_png(plt))
                })
        # 饼图
        if pie[0]:
            for x in pie:
                plt.pie(table_data[x].value_counts(), labels=table_data[x].value_counts().index, autopct="%1.1f%%", shadow=True)
                # 显示纵轴标签
                # plt.ylabel("frequency")
                plt.xlabel("{}".format(x))
                # 显示图标题
                # plt.title("{} - frequency distribution histogram".format(x))
                res.append({
                    "title": "{} - 饼图".format(x),
                    "base64": "{}".format(plot_and_output_base64_png(plt))
                })
        # 矩形图
        if pairPlot[0]:
            sns.pairplot(table_data[pairPlot])
            res.append({
                "title": "特征两两散点图",
                "base64": "{}".format(plot_and_output_base64_png(plt))
            })
        # 相关系数表
        if heatMap[0]:
            corr = table_data[heatMap].corr()
            sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                        linewidths=0.2, cmap="YlGnBu", annot=True)
            # plt.title("Correlation between variables")
            res.append({
                "title": "相关系数热度图",
                "base64": "{}".format(plot_and_output_base64_png(plt))
            })
        # 因变量和自变量的相关系数图
        if yCorr and yCorr["X"][0] and yCorr["Y"][0]:
            corr = table_data[yCorr["X"] + yCorr["Y"]].corr()
            corr[yCorr["Y"][0]].sort_values(ascending=False)[1:].plot(kind='bar')
            plt.ylabel("{}".format(yCorr["Y"][0]))
            # plt.title("Correlations between y and x")
            res.append({
                "title": "因变量和各自变量的相关系数图",
                "base64": "{}".format(plot_and_output_base64_png(plt))
            })
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok"}
        return jsonify(response_data)
    except Exception as e:
        log.exception("Exception Logged")
        response_data = {"data": "", "code": "500", "msg": "{}".format(e.args)}
        # raise e
        return jsonify(response_data)


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, port=5000)
