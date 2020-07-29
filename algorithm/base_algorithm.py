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
from flask import request
from sklearn.model_selection import train_test_split
import os
import joblib
import time
import seaborn as sns
import scipy.stats as stats
from pylab import *
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
                raise ValueError("data must be 2-d list")
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
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        res = []
        try:
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
        except Exception as e:
            log.error(e)
            res = []
        return res

    def show_regression_result(self, x, y, model, options=[]):
        """
        回归模型拟合效果展示
        :param x: 特征列
        :param y: 标签列
        :param model: 已经训练好的回归模型
        :param options: 可选参数，控制输出结果["coff", "independence", "resid_normal"]
        :return: 给前端的结果
        """
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        res = []
        # 拟合优度
        res.append({
            "title": "拟合优度",
            "data": str(model.summary().tables[0])
        })

        # 系数解读
        if "coff" in options:
            res.append({
                "title": "系数解读",
                "data": str(model.summary().tables[1])
            })

        # 独立性检验
        if "independence" in options:
            res.append({
                "title": "独立性检验",
                "data": str(model.summary().tables[2])
            })

        # 残差正态性检验
        if "resid_normal" in options:
            sns.distplot(a=model.resid,
                         bins=10,
                         fit=stats.norm,
                         norm_hist=True,
                         hist_kws={'color': 'green', 'edgecolor': 'black'},
                         kde_kws={'color': 'black', 'linestyle': '--', 'label': '核密度曲线'},
                         fit_kws={'color': 'red', 'linestyle': ':', 'label': '正态密度曲线'}
                         )
            plt.legend()
            res.append({
                "title": "残差正态性检验",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 残差pp图
        if "pp" in options:
            pp_qq_plot = sm.ProbPlot(model.resid)
            pp_qq_plot.ppplot(line='45')
            plt.title('P-P图')
            res.append({
                "title": "残差pp图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 残差qq图
        if "qq" in options:
            pp_qq_plot = sm.ProbPlot(model.resid)
            pp_qq_plot.qqplot(line='q')
            plt.title('Q-Q图')
            res.append({
                "title": "残差qq图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 标准化残差与预测值之间的散点图(验证残差的方差齐性)
        if "var" in options:
            plt.scatter(model.predict(), (model.resid - model.resid.mean()) / model.resid.std())
            plt.xlabel('预测值')
            plt.ylabel('标准化残差')
            plt.title('方差齐性检验')
            # 添加水平参考线
            plt.axhline(y=0, color='r', linewidth=2)
            res.append({
                "title": "方差齐性检验",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 多重共线性检验
        if len(x.columns) > 1 and "vif" in options:
            X = sm.add_constant(x)
            vif = pd.DataFrame()
            vif['features'] = X.columns
            vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            res.append(self.transform_table_data_to_html(
                {
                    "title": "多重共线性检验",
                    "row": vif['features'].values.tolist(),
                    "col": ["VIF Factor"],
                    "data": [vif["VIF Factor"].values.tolist()],
                    "remarks": "VIF>10，存在多重共线性，>100则变量间存在严重的多重共线性"
                }
            ))

        # 线性相关性检验(留在数据探索里面展示)

        # 异常值检测（帽子矩阵、DFFITS准则、学生化残差、Cook距离）
        if "outliers" in options:
            outliers = model.get_influence()
            # 帽子矩阵
            leverage = outliers.hat_matrix_diag
            # dffits值
            dffits = outliers.dffits[0]
            # 学生化残差
            resid_stu = outliers.resid_studentized_external
            # cook距离
            cook = outliers.cooks_distance[0]
            # 合并各种异常值检验的统计量值
            """

            """
            contatl = pd.concat([pd.Series(leverage, name='leverage'),
                                 pd.Series(dffits, name='dffits'),
                                 pd.Series(resid_stu, name='resid_stu'),
                                 pd.Series(cook, name='cook')
                                 ], axis=1)

            x.index = range(x.shape[0])
            profit_outliers = pd.concat([x, contatl], axis=1)
            res.append(self.transform_table_data_to_html(
                {
                    "title": "异常值检测",
                    "row": profit_outliers.index.tolist(),
                    "col": profit_outliers.columns.tolist(),
                    "data": profit_outliers.values.tolist(),
                    "remarks": "当高杠杆值点（或帽子矩阵）大于2(p+1)/n时，则认为该样本点可能存在异常（其中p为自变量的个数，n为观测的个数）；当DFFITS统计值大于2sqrt((p+1)/n)时，则认为该样本点可能存在异常；当学生化残差的绝对值大于2，则认为该样本点可能存在异常；对于cook距离来说，则没有明确的判断标准，一般来说，值越大则为异常点的可能性就越高；对于covratio值来说，如果一个样本的covratio值离数值1越远，则认为该样本越可能是异常值。"
                }
            ))

        # 预测值与真实值的散点图
        if "pred_y_contrast" in options:
            plt.scatter(model.predict(), y)
            plt.plot([model.predict().min(), model.predict().max()],
                     [y.min(), y.max()], 'r-', linewidth=3)
            plt.xlabel('预测值')
            plt.ylabel('实际值')
            plt.title('预测值与真实值对比散点图')
            res.append({
                "title": "预测值与真实值对比散点图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })
        return res

    # 算法输出结果
    def algorithm_show_result(self, model, x, y, options=[], method=None):
        res = []
        try:
            if method == "classifier":
                # 分类测试集结果
                res.extend(self.show_classifier_results(x, y, model, options=options))
            if method == "regression":
                res.extend(self.show_regression_result(x, y, model, options=options))
            if method == "cluster":
                pass
        except Exception as e:
            log.error(e)
            raise e
        return res

    # 初始化数据库读取sql语句
    @staticmethod
    def get_dataframe_from_mysql(sql_sentence, host=None, port=None, user=None, password=None, database=None):
        conn = pymysql.connect(host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com', port=3306, user='yzkj',
                               password='yzkj2020@', database='sophia_data', charset='utf8')
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