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
from util import get_dataframe_from_mysql, transform_h_table_data_to_v, transform_table_data_to_html, \
    exec_sql, format_data, transform_v_table_data_to_h
from flask.json import JSONEncoder as _JSONEncoder
from anova_one_way import normal_test, levene_test, anova_analysis, multiple_test, anova_one_way_describe_info
from anova_all_way import anova_all_way_describe_info, normal_test_all, levene_test_all, anova_analysis_multivariate, \
    level_info
from t_single import t_single_analysis, t_single_describe_info
from t_two_independent import t_two_independent_analysis, t_two_independent_describe_info
from t_two_paried import pearsonr_test, t_two_paired_describe_info, t_two_pair_analysis
from nonparametric_two_independent import wilcoxon_ranksums_test, mannwhitneyu_test, \
    nonparam_two_independent_describe_info
from nonparametric_multi_independent import kruskal_test, median_test, nonparam_multi_independent_describe_info
from two_independent_MWU import Mann_Whitney_U_describe, Mann_Whitney_U_test
from more_independent_KWH import Kruskal_Wallis_H_describe, Kruskal_Wallis_H_test
from nonparametric_two_pair import Wilcoxon_test, Wilcoxon_describe
from crosstable_chi import cross_chi2

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
        elif isinstance(obj, frozenset):
            return list(obj)
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


# ================================ 查看数据 ==============================
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
        data = exec_sql(table_name, X, Y)
        res_data = {
            "title": "查看数据",
            "row": data.index.values.tolist(),
            "col": data.columns.values.tolist(),
            "data": data.values.tolist()
        }
        response_data = {"res": res_data,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e.args)
        # raise e
        return jsonify({"data": "", "code": "500", "msg": e.args[0]})


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
    data = exec_sql(table_name, X, Y)
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
            normal_res = transform_table_data_to_html(normal_test(every_level_data_index, every_level_data, alpha),
                                                      col0="因子水平")
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
        # raise e
        return jsonify({"data": "", "code": "500", "msg": e.args[0]})


# ================================ 多因素方差分析 ==============================
@app.route('/statistic/anovaAllWay', methods=['POST', 'GET'])
def anova_all_way():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
        "analysis_options": ["normal", "variances", "multiple"]
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
        table_direction = request_data['table_direction']
        analysis_options = request_data.get("analysis_options", [])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    data = exec_sql(table_name, X, Y)
    log.info("输入数据大小:{}".format(len(data)))
    try:
        if table_direction == "v":
            data[Y[0]] = data[Y[0]].astype("float16")
            # every_level_data_index = [d for d in data[X[0]].unique()]
            # every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
        elif table_direction == "h":
            # every_level_data_index = X
            # every_level_data = [data[l].astype("float16") for l in X]
            data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        res = []
        # 主体间因子
        res.append(level_info(data, X))
        # 描述性统计分析
        res.append(anova_all_way_describe_info(data, X, Y))
        if "normal" in analysis_options:
            res.append(normal_test_all(data, X, alpha=alpha))
        if "variances" in analysis_options:
            res.append(transform_table_data_to_html(levene_test_all(data, X, alpha=alpha)))
        # 多因素方差分析
        res.append(transform_table_data_to_html(anova_analysis_multivariate(data, X, Y)))
        # todo:稍后加
        # 多重比较
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        # raise e
        return jsonify({"data": "", "code": "500", "msg": e.args[0]})


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
    data = exec_sql(table_name, X)
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
        t_single_res = transform_table_data_to_html(
            t_single_analysis(data[X[0]].astype("float16"), data_mean, X, alpha=alpha), col0="检验值={}".format(data_mean))
        res.append(t_single_res)
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        # raise e
        return jsonify({"data": "error", "code": "500", "msg": e.args[0]})


# ================================ 独立样本t检验 ==============================
@app.route('/statistic/tTwoIndependent', methods=['POST', 'GET'])
def t_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量
        "Y": ["y"], # list,因变量
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
        "analysis_options": ["normal", "variances"]
    }
    :return:
    """
    log.info('t_two_independent_init...')
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
    data = exec_sql(table_name, X, Y)
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
        # 描述统计分析
        res.append(transform_table_data_to_html(t_two_independent_describe_info(data, X, Y)))
        # 正态性检验
        if "normal" in analysis_options:
            res.append(transform_table_data_to_html(normal_test(every_level_data_index, every_level_data, alpha)))
        # 方差齐性检验
        if "variances" in analysis_options:
            res.append(transform_table_data_to_html(levene_test(*every_level_data, alpha=alpha)))
        # 独立样本T检验
        res.append(transform_table_data_to_html(
            t_two_independent_analysis(every_level_data[0], every_level_data[1], alpha=alpha)))
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        # raise e
        return jsonify({"data": "", "code": "500", "msg": e.args[0]})


# ================================ 配对样本t检验 ==============================
@app.route('/statistic/tTwoPair', methods=['POST', 'GET'])
def t_two_pair():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
        "alpha": "0.05", # str,置信区间百分比
        "table_direction": "", str,表格方向,水平方向为h,竖直方向为v
        "analysis_options": ["normal", "pearsonr"]
    }
    :return:
    """
    log.info('t_two_pair_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        table_direction = request_data['table_direction']
        alpha = float(request_data['alpha'])
        analysis_options = request_data.get("analysis_options", [])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    data = exec_sql(table_name, X, Y)
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
        if len(every_level_data_index) > 2:
            raise ValueError("自变量的水平必须是2个")
        res = []
        # 描述性统计分析
        res.append(transform_table_data_to_html(t_two_paired_describe_info(data, X, Y)))
        if "pearsonr" in analysis_options:
            res.append(transform_table_data_to_html(
                pearsonr_test(*every_level_data, index=every_level_data_index, alpha=alpha)))
        if "normal" in analysis_options:
            res.append(transform_table_data_to_html(normal_test(every_level_data_index, every_level_data, alpha)))
        res.append(transform_table_data_to_html(
            t_two_pair_analysis(*every_level_data, index=every_level_data_index, alpha=alpha), col0="配对差值"))
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        # raise e
        return jsonify({"data": "", "code": "500", "msg": e.args[0]})


# ================================ 两独立样本非参数检验 Mann_Whitney_U ==============================
@app.route('/statistic/nonparametricTwoIndependent', methods=['POST', 'GET'])
def nonparametric_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
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
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    data = exec_sql(table_name, X, Y)
    log.info("输入数据大小:{}".format(len(data)))

    try:
        if table_direction == "v":
            every_level_data_index = [d for d in data[X[0]].unique()]
            # every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
            data, X = transform_v_table_data_to_h(data, X, Y)
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            # data, X, Y = transform_h_table_data_to_v(data, X) # 水平的数据，这里不用转
        else:
            raise ValueError("table direction must be h or v")
        if len(every_level_data_index) > 2:
            raise ValueError("自变量的水平必须是2个")

        # 描述性统计
        res = []
        data_info = transform_table_data_to_html(Mann_Whitney_U_describe(data, X))
        res.append(data_info)

        # Mann-Whitney U 检验
        Mann_Whitney_U_res = transform_table_data_to_html(Mann_Whitney_U_test(data, X))
        res.append(Mann_Whitney_U_res)
        response_data = {"res": res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e
        # return jsonify({"data": "", "code": "500", "msg": e.args[0]})


# ================================ 多个独立样本非参数检验 Kruskal-Wallis H 检验==============================
@app.route('/statistic/nonparametricMultiIndependent', methods=['POST', 'GET'])
def results_nonparametric_multi_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        "Y": ["y"], # list,因变量,当表格方向为v是使用
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
        # alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # 从数据库拿数据
    data = exec_sql(table_name, X, Y)
    log.info("输入数据大小:{}".format(len(data)))

    try:
        if table_direction == "v":
            every_level_data_index = [d for d in data[X[0]].unique()]
            # every_level_data = [data[data[X[0]] == d][Y[0]].astype("float16") for d in data[X[0]].unique()]
            data, X = transform_v_table_data_to_h(data, X, Y)
        elif table_direction == "h":
            every_level_data_index = X
            every_level_data = [data[l].astype("float16") for l in X]
            # data, X, Y = transform_h_table_data_to_v(data, X)
        else:
            raise ValueError("table direction must be h or v")
        if len(every_level_data_index) < 2:
            raise ValueError("多个独立样本非参数检验，自变量的水平至少是2个")

        # 描述性统计
        res = []
        data_info = transform_table_data_to_html(Kruskal_Wallis_H_describe(data, X))
        res.append(data_info)
        log.info("描述性统计分析完成")

        # Kruska-Wallis H 检验
        Kruskal_Wallis_H_res = Kruskal_Wallis_H_test(data, X)
        res.append(Kruskal_Wallis_H_res)
        log.info("Kruska-Wallis H 检验完成")

        response_data = {"res": res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e


# ================================ 两配对样本非参数检验  Wilcoxon 符号秩检验 ==============================
@app.route('/statistic/nonparametricTwoPair', methods=['POST', 'GET'])
def results_nonparametric_two_independent():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
        # "table_direction": "h", str,表格方向,水平方向为h,竖直方向为v
    }
    :return:
    """
    log.info('nonparametric_two_pair_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        # Y = request_data['Y']
        # table_direction = request_data['table_direction']
        # alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    # assert isinstance([X, Y], list)
    assert isinstance([X], list)
    if len(X) > 2:
        raise ValueError("只支持一列数据或两列数据")
    # 从数据库拿数据
    # data = exec_sql(table_name, X, Y)
    data = exec_sql(table_name, X)
    log.info("输入数据大小:{}".format(len(data)))

    try:

        # 描述性统计
        res = []
        data_info = transform_table_data_to_html(Wilcoxon_describe(data, X))
        res.append(data_info)
        log.info("描述性统计分析完成")

        # Wilcoxon 符号秩检验
        Wilcoxon_res = transform_table_data_to_html(Wilcoxon_test(data, X))
        res.append(Wilcoxon_res)
        log.info("Wilcoxon 符号秩检验完成")

        response_data = {"res": res,
                         "data_info": data_info,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e


# ================================ 交叉表 及卡方检验 ==============================
@app.route('/statistic/crosstable', methods=['POST', 'GET'])
def results_crosstable():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "X": ["x1"], # list,自变量，行
        "Y": ["y"], # list,因变量，列

    }
    :return:
    """
    log.info('crosstable_get_results_init...')
    request_data = init_route()
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        Y = request_data['Y']
        # table_direction = request_data['table_direction']
        # alpha = float(request_data['alpha'])
    except Exception as e:
        log.info(e)
        raise e
    assert isinstance([X, Y], list)
    # assert isinstance([X], list)
    # 从数据库拿数据
    data = exec_sql(table_name, X, Y)
    # data = exec_sql(table_name, X)
    log.info("输入数据大小:{}".format(len(data)))

    try:
        index = data[X[0]]
        columns = data[Y[0]]
        res = cross_chi2(index, columns)

        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.error(e)
        raise e


# ================================ 关联规则Apriori/fpgrowth ==============================
@app.route('/statistic/apriori', methods=['POST', 'GET'])
def apriori():
    """
    接口请求参数:{
        "table_name": "apriori_test",  # str,数据库表名
        "X": ["x0", "x1", "x2", "x3", "x4", "x5"],  # list,自变量
        "alg": "fpgrowth',  # str,关联规则算法选择["apriori", "fpgrowth"] ==》【默认值：fpgrowth】
        "dataconvert": True,  # bool,是否需要数据转换 ==》【默认值：True】
        "minSupport": "0.05",  # str,最小支持度 ==》【默认值："0.05"】
        "max_len": "2",  # 频繁项集最大长度 ==》【默认值：None】
        "metrics": "confidence",  # 关联规则评价指标["support", "confidence", "lift", "leverage", "conviction"] ==》【默认值：confidence】
        "min_threshold": "0.8",  # 关联规则评价指标最小值 ==》【默认值："0.8"】
    }
    :return:
    """
    log.info('Apriori_init...')
    request_data = init_route()
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.frequent_patterns import association_rules
    except:
        raise ImportError("cannot import mlxtend")
    try:
        table_name = request_data['table_name']
        X = request_data['X']
        alg = request_data['alg']
        dataconvert = request_data['dataconvert']
        min_support = float(request_data['minSupport'])
        max_len = int(request_data['max_len'])
        metrics = request_data['metrics']
        min_threshold = float(request_data['min_threshold'])
    except Exception as e:
        log.info(e)
        raise e
    try:
        table_data = exec_sql(table_name, X)
        data = table_data.values.tolist()
        if dataconvert:
            trans = TransactionEncoder()
            data = trans.fit(data).transform(data)
            data = pd.DataFrame(data, columns=trans.columns_)
            data.drop([""], axis=1, inplace=True)
        if alg == "apriori":
            frequent_itemsets = apriori(data, min_support=min_support, max_len=max_len, use_colnames=True)
        elif alg == "fpgrowth":
            frequent_itemsets = fpgrowth(data, min_support=min_support, max_len=max_len, use_colnames=True)
        else:
            raise ValueError("input Association rules:{} is not support".format(alg))
        rules = association_rules(frequent_itemsets, metric=metrics, min_threshold=min_threshold)
        res = [
            transform_table_data_to_html({
                "title": "频繁项集结果",
                "row": frequent_itemsets.index.tolist(),
                "col": frequent_itemsets.columns.tolist(),
                "data": frequent_itemsets.values.tolist(),
            }),
            transform_table_data_to_html({
                "title": "关联规则结果",
                "row": rules.index.tolist(),
                "col": rules.columns.tolist(),
                "data": rules.values.tolist(),
            })
        ]
        response_data = {"res": res,
                         "code": "200",
                         "msg": "ok!"}
        return jsonify(response_data)
    except Exception as e:
        log.exception(e)
        return jsonify({"code": "500", "res": "", "msg": "{}".format(e.args)})


if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0", debug=True, port=5000)
