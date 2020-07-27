# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : base_algorithm

Description : 所有算法的基类

Author : leiliang

Date : 2020/7/25 12:24 下午

--------------------------------------------------------

"""
import base64
from io import BytesIO
import logging
import pymysql
from sklearn import metrics
import numpy as np
import pandas as pd
from flask import request, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import joblib
import time
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class BaseAlgorithm(object):
    def __init__(self):
        self.web_data = self.get_web_data()

    def train(self):
        raise NotImplementedError("请继承此基类并在子类中实现该方法")

    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_config_from_web(self):
        raise NotImplementedError("请继承此基类并在子类中实现该方法")

    # 划分测试集和训练集
    @staticmethod
    def split_data(data, config):
        try:
            data_x = data[config["X"]]
            data_y = data[config["Y"][0]]
            # 数据分割
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                                random_state=config['randomState'],
                                                                test_size=config['rate'])
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise e

    # 读取模型
    @staticmethod
    def load_model(name):
        try:
            model_name_list = os.listdir("./model/{}".format(name))
            model_name_list.sort()
            latest_model_path = os.path.join("./model/{}".format(name), model_name_list[-1])
            test_model = joblib.load(latest_model_path)
            return test_model
        except Exception as e:
            raise e

    # 保存模型
    @staticmethod
    def save_model(model, model_name):
        try:
            # 保存模型
            if not os.path.exists("./model/{}/".format(model_name)):
                os.mkdir("./model/{}/".format(model_name))
            save_path = "./model/{}/{}.pkl".format(model_name, time.strftime("%y-%m-%d-%H-%M-%S", time.localtime()))
            joblib.dump(model, save_path)
            log.info("save model in {}".format(save_path))
        except Exception as e:
            raise e

    # 转换输出的表格数据让前端识别并显示
    @staticmethod
    def transform_table_data_to_html(data: dict, col0=""):
        data["col"].insert(0, col0)
        for idx, (index, row) in enumerate(zip(data["row"], data["data"])):
            if not isinstance(data["data"][idx], list):
                data["data"][idx] = list(data["data"][idx])
            data["data"][idx].insert(0, str(index))
        if "row" in data:
            del data["row"]
        return data

    # format dataframe
    @staticmethod
    def format_dataframe(data, config):
        for key, value in config.items():
            data[key] = data[key].map(lambda x: format(x, value))
        return data

    # 分类评估报告转表格数据输出给前端
    @staticmethod
    def report_to_table_data(report):
        col = ["精确率", "召回率", "F1", "样本数"]
        row = []
        data = []
        for row_data in report.split("\n\n")[1].split("\n"):
            row_data = [r for r in row_data.split(" ") if r]
            row.append(row_data[0])
            data.append(row_data[1:])
        for row_data in report.split("\n\n")[2].split("\n"):
            if not row_data:
                continue
            row_data = [r for r in row_data.split(" ") if r]
            if row_data[0] == "accuracy":
                row.append(row_data[0])
                data.append(["", ""] + row_data[1:])
            else:
                row.append(row_data[0] + " " + row_data[1])
                data.append(row_data[2:])
        return {
            "row": row,
            "col": col,
            "data": data,
            "title": "分类报告:precision/recall/F1/分类个数",
            "remarks": "accuracy:准确率(正负样本总的正确分类的比率)，"
                       "macro avg:宏平均(所有类的精确率、召回率、F1的平均值), "
                       "weighted avg:加权平均基于样本个数加权平均精确率、召回率、F1)"
        }

    # matplotlib作图写入内存并输出base64格式供前端调用
    @staticmethod
    def plot_and_output_base64_png(plot):
        # 写入内存
        save_file = BytesIO()
        plot.savefig(save_file, format='png')

        # 转换base64并以utf8格式输出
        save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        return save_file_base64

    # 机器学习模型分类效果展示
    def show_classifier_results(self, x, y, model, options=[]):
        """
        机器学习模型分类效果展示
        :param x: 特征列
        :param y: 标签列
        :param model: 已经训练好的分类模型
        :param options: 可选参数，控制输出结果["report", "matrix", "roc"]
        :return: 给前端的结果
        """
        res = []
        # 输出结果展示
        y_predict = model.predict(x)
        # y_predict_proba = model.predict_proba(x)

        # 分类评估报告输出表格数据,默认展示
        # accuracy_score = metrics.accuracy_score(y, y_predict)
        # precision_score = metrics.precision_score(y, y_predict)
        # recall_score = metrics.recall_score(y, y_predict)
        # f1_score = metrics.f1_score(y, y_predict)
        report = metrics.classification_report(y, y_predict, target_names=model.classes_.tolist())
        res.append(self.transform_table_data_to_html(self.report_to_table_data(report)))

        # 输出混淆矩阵图片
        if "matrix" in options:
            metrics.plot_confusion_matrix(model, x, y)
            plt.title("confusion_matrix")
            res.append({
                "title": "混淆矩阵",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 输出roc、auc图片
        if "roc" in options:
            metrics.plot_roc_curve(model, x, y)
            plt.title("roc-auc")
            res.append({
                "title": "ROC曲线和auc",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        return res

    # 算法输出结果
    def algorithm_show_result(self, model, x, y, options=[], method=[]):
        res = []
        if "classifier" in method:
            # 分类测试集结果
            res.extend(self.show_classifier_results(x, y, model, options=options))

        if "regression" in method:
            # 拟合优度结果（回归算法才有）
            try:
                import statsmodels.api as sm
            except:
                raise ImportError("statsmodels.api cannot import")
            try:
                x = x.astype(float)
                y = y.astype(float)
                x = sm.add_constant(x)
                logit_stats_res = sm.Logit(y, x).fit()
                # 拟合优度
                if "r2" in options:
                    res.append({
                        "title": "逻辑回归统计分析结果",
                        "data": str(logit_stats_res.summary().tables[0])
                    })
                # 系数解读
                if "coff" in options:
                    res.append({
                        "title": "逻辑回归系数解读",
                        "data": str(logit_stats_res.summary().tables[1])
                    })
            except Exception as e:
                log.error("statsmodels analysis error")
                # raise e
        if "cluster" in method:
            pass
        return res

    # 初始化数据库读取sql语句
    @staticmethod
    def get_dataframe_from_mysql(sql_sentence, host=None, port=None, user=None, password=None, database=None):
        conn = pymysql.connect(host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com', port=3306, user='yzkj',
                               password='yzkj2020@', database='sophia_manager', charset='utf8')
        try:
            df = pd.read_sql(sql_sentence, conn)
            return df
        except Exception as e:
            raise e

    # 根据sql获取数据
    def exec_sql(self, table_name, X=None, Y=None):
        # 从数据库拿数据
        try:
            if not Y or Y[0] == "":
                sql_sentence = "select {} from {};".format(",".join(X), "`" + table_name + "`")
            else:
                sql_sentence = "select {} from {};".format(",".join(X + Y), "`" + table_name + "`")
            data = self.get_dataframe_from_mysql(sql_sentence)
            return data
        except Exception as e:
            log.info(e.args)
            raise e

    # 获取从前端传来的参数
    @staticmethod
    def get_web_data():
        try:
            request_data = request.json
            log.info("receive request :{}".format(request_data))
        except Exception as e:
            log.info(e)
            raise e
        return request_data
