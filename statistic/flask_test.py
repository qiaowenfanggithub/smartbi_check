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
import json

if __name__ == '__main__':
    # data = pd.read_csv("./data/PimaIndiansdiabetes.csv")
    # data = pd.read_excel("./data/buy-computer.xlsx")
    my_session = requests.session()
    # ======================= 单因素方差分析-查看数据 =============================
    # kwargs = {
    #     "table_name": "t_single",  # str,数据库表名
    #     "X": ["value"],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/statistic/checkData', json=kwargs, timeout=30)
    # res = my_session.post(url='https://www.yzsmart.top/statistic/checkData', json=kwargs, timeout=30)

    # ======================= 单因素方差分析-结果 =============================
    # kwargs = {
    #     "table_name": "anova_one_way",  # str,数据库表名
    #     "X": ["level"],  # list,自变量
    #     "Y": ["value"],  # list,因变量
    #     "alpha": "0.06",  # str,置信区间百分比
    #     "table_direction": "v",  # 表格方向，一般为竖向，即有一列是分类变量
    #     "analysis_options": ["normal", "variances", "multiple"],
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/statistic/anovaOneWay', json=kwargs, timeout=30)
    # res = my_session.post(url='http://121.42.242.214:8099/anovaOneWay', json=kwargs, timeout=30)
    # res = my_session.post(url='https://www.yzsmart.top/statistic/anovaOneWay', json=kwargs, timeout=30)

    # ======================= 多因素方差分析-检验 =============================
    # kwargs = {
    #     "table_name": "anova_all_way",  # str,数据库表名
    #     "X": ["培训前成绩等级", "培训方法"],  # list,自变量
    #     "Y": ["成绩"],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/anovaAllWay/test', json=kwargs, timeout=30)

    # ======================= 多因素方差分析-结果 =============================
    # kwargs = {
    #     "table_name": "anova_all_way",  # str,数据库表名
    #     "X": ["培训前成绩等级", "培训方法"],  # list,自变量
    #     "Y": ["成绩"],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/anovaAllWay/results', json=kwargs, timeout=30)

    # ======================= 单样本t检验-检验 =============================
    kwargs = {
        "table_name": "t_single",  # str,数据库表名
        "X": ["value"],  # list,自变量
        "mean": "80",  # list,自变量
        "alpha": "0.05",  # str,置信区间百分比
        "analysis_options": ["normal"],
    }
    res = my_session.post(url='http://127.0.0.1:5000/statistic/tSingle', json=kwargs, timeout=30)

    # ======================= 单样本t检验-结果 =============================
    # kwargs = {
    #     "table_name": "t_single",  # str,数据库表名
    #     "X": ["value"],  # list,自变量
    #     "mean": "80",  # list,自变量
    #     "alpha": "0.05",  # str,置信区间百分比
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/tSingle/results', json=kwargs, timeout=30)

    # ======================= 独立样本t检验-检验 =============================
    # kwargs = {
    #     "table_name": "t_two_independent",  # str,数据库表名
    #     "X": ["level"],  # list,自变量
    #     "Y": ["value"],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "v",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/tTwoIndependent/test', json=kwargs, timeout=30)

    # ======================= 独立样本t检验-结果 =============================
    # kwargs = {
    #     "table_name": "t_two_independent",  # str,数据库表名
    #     "X": ["level"],  # list,自变量
    #     "Y": ["value"],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "v",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/tTwoIndependent/results', json=kwargs, timeout=30)

    # ======================= 配对样本t检验-检验 =============================
    # kwargs = {
    #     "table_name": "t_two_pair",  # str,数据库表名
    #     "X": ["value1", "value2"],  # list,自变量
    #     "Y": [""],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "h",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/tTwoPair/test', json=kwargs, timeout=30)

    # ======================= 配对样本t检验-结果 =============================
    # kwargs = {
    #     "table_name": "t_two_pair",  # str,数据库表名
    #     "X": ["value1", "value2"],  # list,自变量
    #     "Y": [""],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "h",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/tTwoPair/results', json=kwargs, timeout=30)

    # ======================= 两独立样本非参数检验-结果 =============================
    # kwargs = {
    #     "table_name": "nonparametric_two_independent",  # str,数据库表名
    #     "X": ["x1", "x2"],  # list,自变量
    #     "Y": [""],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "h",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/nonparametricTwoIndependent/results', json=kwargs, timeout=30)

    # ======================= 两配对样本非参数检验-结果 =============================
    # kwargs = {
    #     "table_name": "nonparametric_two_pair_diff",  # str,数据库表名
    #     "X": ["x1"],  # list,自变量
    #     "Y": [""],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "h",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/nonparametricTwoPair/results', json=kwargs, timeout=30)

    # ======================= 多个独立样本非参数检验-结果 =============================
    # kwargs = {
    #     "table_name": "nonparametric_multi_independent",  # str,数据库表名
    #     "X": ["level"],  # list,自变量
    #     "Y": ["value"],  # list,因变量
    #     "alpha": "0.05",  # str,置信区间百分比
    #     "table_direction": "v",  # 表格方向，一般为竖向，即有一列是分类变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/nonparametricMultiIndependent/results', json=kwargs, timeout=30)

    # print(json.loads(res.text, encoding="utf-8"))
    print(res.text)