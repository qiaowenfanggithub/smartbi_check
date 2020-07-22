# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : single_samples_t

Description : 统计分析平台

Author : leiliang

Date : 2020/7/1 3:30 下午

--------------------------------------------------------

"""
from __future__ import print_function
from flask import Flask, request, jsonify
import logging
import time
import json
import numpy as np
import pandas as pd
from flask_cors import *
from utils import get_dataframe_from_mysql, transform_h_table_data_to_v, transform_table_data_to_html
from flask.json import JSONEncoder as _JSONEncoder
from anova_one_way import normal_test, levene_test, anova_analysis, multiple_test, anova_one_way_describe_info
from anova_all_way import anova_analysis_multivariate, multiple_test_multivariate, anova_all_way_describe_info
from t_single import t_single_analysis, t_single_describe_info
from t_two_independent import t_two_independent_analysis, t_two_independent_describe_info
from t_two_paried import pearsonr_test, t_two_paired_describe_info, t_two_pair_analysis
from nonparametric_two_independent import wilcoxon_ranksums_test, mannwhitneyu_test, \
    nonparam_two_independent_describe_info
from nonparametric_two_pair import mannwhitneyu_test_with_diff, nonparam_two_paired_describe_info
from nonparametric_multi_independent import kruskal_test, median_test, nonparam_multi_independent_describe_info

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


def init_route():
    log_file = "statistic.log"
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

# ================================ 单因素方差分析-查看数据 ==============================
@app.route('/statistic/checkData', methods=['POST', 'GET'])
def check_data():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名-数据处理之后的数据
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
    }
    :return:
    """
    log.info('anova_one_way_check_data_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data.get('Y', [])
    except Exception as e:
        log.info(e)
        raise e
    # 从数据库拿数据：
    try:
        if not Y or (len(Y) == 1 and Y[0] == ""):
            sql_sentence = "select {} from {};".format(",".join(X), "`" + table_name + "`")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), "`" + table_name + "`")
        data = get_dataframe_from_mysql(sql_sentence)
        res_data = {
            "title": "单因素方差分析-查看数据",
            "row": data.index.values.tolist(),
            "col": data.columns.values.tolist(),
            "data": data.values.tolist()
        }
        response_data = {"res": res_data,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.info(e.args)
        raise e


# ================================ 单因素方差分析 ==============================
@app.route('/statistic/anovaOneWay', methods=['POST', 'GET'])
def anova_one_way():
    """
    接口请求参数:{
        "table_name_ori": "" # str,数据库表名-数据预处理之前的数据
        "table_name": "" # str,数据库表名-数据处理之后的数据
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
        "analysis_options": ["normal", "variances", "multiple"]
    }
    :return:
    """
    log.info('anova_one_way_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        alpha = float(request_data['alpha'])
        table_direction = request_data['table_direction']
        analysis_options = request_data.get("analysis_options", [])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), "`" + table_name + "`")
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), "`" + table_name + "`")
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            data[Y[0]] = data[Y[0]].astype("float16")
            every_level_data_index = [d for d in data[X[0]].unique()]
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        res = []
        # 描述性统计分析
        data_info = transform_table_data_to_html(anova_one_way_describe_info(data, X, Y, alpha=alpha))
        res.append(data_info)
        # 正太分布检验
        if "normal" in analysis_options:
            normal_res = transform_table_data_to_html(normal_test(every_level_data_index, every_level_data, alpha), col0="因子水平")
            res.append(normal_res)
        # 方差齐性检验
        if "variances" in analysis_options:
            equal_variances_res = transform_table_data_to_html(levene_test(*every_level_data, alpha=alpha))
            res.append(equal_variances_res)
        # 方差分析
        anova_res = transform_table_data_to_html(anova_analysis(data, X[0], Y[0], alpha=alpha))
        res.append(anova_res)
        # 多重比较
        if "multiple" in analysis_options:
            multiple_res = multiple_test(data, X, Y, alpha=alpha)
            res.append(multiple_res)
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 多因素方差分析 ==============================
@app.route('/anovaAllWay/test', methods=['POST', 'GET'])
def test_anova_all_way():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('anova_all_way_test_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        alpha = float(request_data['alpha'])
        # todo:暂时只支持竖向表格
        table_direction = request_data['table_direction']
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    data_info = anova_all_way_describe_info(data, X, Y)
    try:
        normal_res_list = []
        equal_variances_res_list = []
        for i in range(len(X)):
            every_level_data_index = [d for d in data[X[i]].unique()]
            every_level_data = [data[data[X[i]] == d][Y[0]].astype("float16") for d in data[X[i]].unique()]
            """
                正太分布检验
            """
            normal_res = normal_test(every_level_data_index, every_level_data, alpha)
            normal_res_list.append((X[i], normal_res))
            """
                方差齐性检验
            """
            equal_variances_res = levene_test(*every_level_data, alpha=alpha)
            equal_variances_res_list.append((X[i], equal_variances_res))
        response_data = {"data_info": data_info,
                         "normalTest": normal_res_list,
                         "equalVariancesTest": equal_variances_res_list,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


@app.route('/anovaAllWay/results', methods=['POST', 'GET'])
def results_anova_all_way():
    """
    单因素方差分析结果展示
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('anova_all_way_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        alpha = float(request_data['alpha'])
        # todo:暂时只支持竖向表格
        table_direction = request_data['table_direction']
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    data[Y[0]] = data[Y[0]].astype("float16")
    """
        多因素方差分析
    """
    anova_res_multivariate = anova_analysis_multivariate(data, X, Y)
    """
        多重比较
    """
    multiple_res_multivariate = multiple_test_multivariate(data, X, Y, alpha=alpha)
    response_data = {"anova_res": anova_res_multivariate,
                     "multiple_res": multiple_res_multivariate,
                     "code": "200",
                     "msg": "ok!"}
    return jsonify(response_data)


# ================================ 单样本t检验 ==============================
@app.route('/statistic/tSingle', methods=['POST', 'GET'])
def t_single():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["value"], # list,自变量
        "alpha": "0.05", # str,置信区间百分比
        "mean": "0", # str,样本均值
        "analysis_options": ["normal"]
    }
    :return:
    """
    log.info('t_single_test_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        alpha = float(request_data['alpha'])
        X = request_data['X']
        data_mean = float(request_data['mean'])
        analysis_options = request_data.get("analysis_options", [])
    except Exception as e:
        log.info(e)
        raise e
    # 从数据库拿数据
    try:
        sql_sentence = "select {} from {};".format(",".join(X), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        res = []
        data[X[0]] = data[X[0]].astype("float16")
        data_info = transform_table_data_to_html(t_single_describe_info(data, X))
        res.append(data_info)
        # 正态性检验
        if "normal" in analysis_options:
            normal_res = transform_table_data_to_html(normal_test([X[0]], [data[X[0]]], alpha=alpha))
            res.append(normal_res)
        # 单样本t检验分析结果
        t_single_res = transform_table_data_to_html(t_single_analysis(data[X[0]].astype("float16"), data_mean, X, alpha=alpha), col0="检验值={}".format(data_mean))
        res.append(t_single_res)
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "error", "code": "500", "msg": e.args})


# ================================ 独立样本t检验 ==============================
@app.route('/tTwoIndependent/test', methods=['POST', 'GET'])
def test_t_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('t_two_independent_test_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        alpha = float(request_data['alpha'])
        table_direction = request_data['table_direction']
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        """
            正太分布检验
        """
        if table_direction == "v":
            every_level_data_index = [d for d in data[X[0]].unique()]
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        data_info = t_two_independent_describe_info(data, X, Y)
        normal_res = normal_test(every_level_data_index, every_level_data, alpha)
        response_data = {"normalTest": normal_res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


@app.route('/tTwoIndependent/results', methods=['POST', 'GET'])
def results_t_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('t_two_independent_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        alpha = float(request_data['alpha'])
        table_direction = request_data['table_direction']
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        """
            两独立样本t检验分析
        """
        if table_direction == "v":
            assert len(data[X[0]].unique()) == 2, "input x must only 2 level"
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data = [data[l].astype("float16") for l in X]
        else:
            raise ValueError("table direction must be h or v")
        t_two_independent_res = t_two_independent_analysis(every_level_data[0], every_level_data[1], alpha=alpha)
        response_data = {"res": t_two_independent_res,
                         "row": X,
                         "col": ["equal_var", "F值", "Significance", "t值", "p值"],
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 配对样本t检验 ==============================
@app.route('/tTwoPair/test', methods=['POST', 'GET'])
def test_t_two_pair():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('t_two_pair_test_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        """
            正太分布检验
        """
        if table_direction == "v":
            assert Y[0] != "", "input Y must not be empty when table direction is v"
            every_level_data_index = [d for d in data[X[0]].unique()]
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        data_info = t_two_paired_describe_info(data, X, Y)
        normal_res = normal_test(every_level_data_index, every_level_data, alpha)
        response_data = {"normalTest": normal_res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


@app.route('/tTwoPair/results', methods=['POST', 'GET'])
def results_t_two_pair():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('t_two_pair_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            assert Y[0] != "", "input Y must not be empty when table direction is v"
            every_level_data_index = [d for d in data[X[0]].unique()]
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
        else:
            raise ValueError("table direction must be h or v")
        pearsonr_value = pearsonr_test(every_level_data, index=every_level_data_index)
        t_two_paired_res = t_two_pair_analysis(*every_level_data, index=every_level_data_index)
        response_data = {"pearsonr_value": pearsonr_value,
                         "t_two_paired_res": t_two_paired_res,
                         "row": X,
                         "col": ["equal_var", "F值", "Significance", "t值", "p值"],
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 两独立样本非参数检验 ==============================
@app.route('/nonparametricTwoIndependent/results', methods=['POST', 'GET'])
def results_nonparametric_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('nonparametric_two_independent_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        data_info = nonparam_two_independent_describe_info(data, X, Y)
        res = [{"mannwhitneyu_test": mannwhitneyu_test(every_level_data[0], every_level_data[1])},
               {"wilcoxon_ranksums": wilcoxon_ranksums_test(every_level_data[0], every_level_data[1])}]
        response_data = {"res": res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 两配对样本非参数检验 ==============================
@app.route('/nonparametricTwoPair/results', methods=['POST', 'GET'])
def results_nonparametric_two_pair():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('nonparametric_two_pair_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        if len(X) == 2:
            mannwhitneyu_test_res = mannwhitneyu_test(every_level_data[0], every_level_data[1])
        elif len(X) == 1:
            mannwhitneyu_test_res = mannwhitneyu_test_with_diff(data[X[0]].astype("float16"))
        else:
            raise ValueError("input X must be 1 or 2")
        # todo:这里应该是要加入秩和检验

        # 描述性统计分析
        data_info = nonparam_two_paired_describe_info(data, X, Y)
        res = [{"mannwhitneyu_test": mannwhitneyu_test_res}]
        response_data = {"res": res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


# ================================ 多个独立样本非参数检验 ==============================
@app.route('/nonparametricMultiIndependent/results', methods=['POST', 'GET'])
def results_nonparametric_multi_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('nonparametric_multi_independent_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    try:
        if len(Y) == 1 and Y[0] == "":
            sql_sentence = "select {} from {};".format(",".join(X), table_name)
        else:
            sql_sentence = "select {} from {};".format(",".join(X + Y), table_name)
        data = get_dataframe_from_mysql(sql_sentence)
    except Exception as e:
        log.info(e.args)
        raise e
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            every_level_data_index = [d for d in data[X[0]].unique()]
            every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        data_info = nonparam_multi_independent_describe_info(data, X, Y)
        kw_res = kruskal_test(*every_level_data)
        median_res = median_test(*every_level_data, col_list=every_level_data_index)
        response_data = {"kw_res": kw_res,
                         "data_info": data_info,
                         "median_res": median_res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args})


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0", debug=True, port=5000)
