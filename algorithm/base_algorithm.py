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
from utils import format_dataframe
from base64_to_png import base64_to_img
import random
import datetime
import json
import uuid

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
            if len(model_name_list) > 20:
                del model_name_list[:len(model_name_list) - 20]
            latest_model_path = os.path.join("./model/{}".format(name), model_name_list[-1])
            test_model = joblib.load(latest_model_path)
            return test_model
        except Exception as e:
            raise e

    # 模型信心入库
    def save_model_into_database(self, model_name, current_time=None):
        try:
            # 模型入库
            if not current_time:
                current_time = datetime.datetime.now()
            userid = "000"
            name = "{}-{}".format(model_name, current_time.strftime("%Y-%m-%d-%H-%M-%S"))
            type = "{}".format(model_name)
            characteristic_column = ",".join(self.config['X'])
            label_column = self.config['Y'][0] if self.config['Y'] else ""
            data_set = self.config['tableName']
            parameter_config = json.dumps(self.config, ensure_ascii=False)
            save_path = "./model/{}/{}.pkl".format(model_name, name)
            result_report = ""
            updatetime = current_time.strftime("%Y-%m-%d %H:%M:%S")
            if not os.path.exists("./model/{}".format(model_name)):
                os.makedirs("./model/{}".format(model_name))
            joblib.dump(self.model, save_path)
            key_list = [["userid", "name", "type", "characteristic_column", "label_column",
                         "data_set", "parameter_config", "save_path", "result_report", "updatetime"]]
            value_list = [[userid, name, type, characteristic_column, label_column,
                           data_set, parameter_config, save_path, result_report, updatetime]]
            sql_list = self.exec_insert_sql("algorithm_model", key_list, value_list)
            log.info("exec sql:{} finish".format(sql_list[0]))
        except Exception as e:
            raise e

    # 从数据库读取模型
    @staticmethod
    def load_model_by_database(algorithm, model):
        try:
            model_path = "./model/{}/{}.pkl".format(algorithm, model)
            test_model = joblib.load(model_path)
            return test_model
        except Exception as e:
            raise e

    # 保存模型
    @staticmethod
    def save_model(model, model_name):
        try:
            # 保存模型
            if not os.path.exists("./model/{}/".format(model_name)):
                os.makedirs("./model/{}/".format(model_name))
            model_name_list = os.listdir("./model/{}".format(model_name))
            model_name_list.sort()
            if len(model_name_list) > 19:
                for m in model_name_list[:len(model_name_list) - 19]:
                    os.remove("./model/{}/{}".format(model_name, m))
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
        plot.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
        plot.rcParams["axes.unicode_minus"] = False
        # 写入内存
        save_file = BytesIO()
        plot.savefig(save_file, format='png')
        # 转换base64并以utf8格式输出
        save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        # debug
        # base64_to_img(save_file_base64)
        plot.close("all")

        # 写入文件
        # tmp_file_name = uuid.uuid4()
        # plot.savefig("./img/{}.png".format(tmp_file_name))
        # with open("./img/{}.png".format(tmp_file_name), "rb") as f:
        #     save_file_base64 = base64.b64encode(f.read()).decode('utf8')

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
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # plt.rcParams['axes.unicode_minus'] = False
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
            if "report" in options:
                report = metrics.classification_report(y, y_predict,
                                                       target_names=["{}".format(s) for s in model.classes_.tolist()])
                # res.append(self.transform_table_data_to_html(self.report_to_table_data(report)))
                res.append({
                    "is_test": False,
                    "title": "分类报告",
                    "str": report.replace("\n", "<br/>")
                })

            # 输出混淆矩阵图片
            if "matrix" in options:
                metrics.plot_confusion_matrix(model, x, y)
                plt.title("confusion matrix")
                res.append({
                    "is_test": False,
                    "title": "混淆矩阵",
                    "base64": "{}".format(self.plot_and_output_base64_png(plt))
                })

            # 输出roc、auc图片
            if "roc" in options:
                metrics.plot_roc_curve(model, x, y)
                plt.title("roc curve")
                res.append({
                    "is_test": False,
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
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # plt.rcParams['axes.unicode_minus'] = False
        res = []
        # 拟合优度
        if "r2" in options:
            res.append({
                "is_test": False,
                "title": "拟合优度",
                "str": str(model.summary().tables[0]).replace("\n", "<br/>")
            })

        # 系数解读
        if "coff" in options:
            res.append({
                "is_test": False,
                "title": "系数解读",
                "str": str(model.summary().tables[1]).replace("\n", "<br/>")
            })

        # 独立性检验
        if "independence" in options:
            res.append({
                "is_test": True,
                "title": "独立性检验",
                "str": str(model.summary().tables[2]).replace("\n", "<br/>")
            })

        # 残差正态性检验
        if "resid_normal" in options:
            sns.distplot(a=model.resid,
                         bins=10,
                         fit=stats.norm,
                         norm_hist=True,
                         hist_kws={'color': 'green', 'edgecolor': 'black'},
                         kde_kws={'color': 'black', 'linestyle': '--', 'label': 'kernel density curve'},
                         fit_kws={'color': 'red', 'linestyle': ':', 'label': 'normal density curve'}
                         )
            plt.legend()
            plt.title("残差正态性检验")
            res.append({
                "is_test": True,
                "title": "残差正态性检验",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 残差pp图
        if "pp" in options:
            pp_qq_plot = sm.ProbPlot(model.resid)
            pp_qq_plot.ppplot(line='45')
            res.append({
                "is_test": True,
                "title": "残差pp图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 残差qq图
        if "qq" in options:
            pp_qq_plot = sm.ProbPlot(model.resid)
            pp_qq_plot.qqplot(line='q')
            res.append({
                "is_test": True,
                "title": "残差qq图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 标准化残差与预测值之间的散点图(验证残差的方差齐性)
        if "var" in options:
            plt.scatter(model.predict(), (model.resid - model.resid.mean()) / model.resid.std())
            plt.xlabel('predict value')
            plt.ylabel('standardized residual ')
            # 添加水平参考线
            plt.axhline(y=0, color='r', linewidth=2)
            res.append({
                "is_test": True,
                "title": "方差齐性检验",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })

        # 多重共线性检验
        if len(x.columns) > 1 and "vif" in options:
            X = sm.add_constant(x)
            vif = pd.DataFrame()
            vif['features'] = X.columns
            vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif = format_dataframe(vif, {"VIF Factor": ".4f"})
            res.append(self.transform_table_data_to_html(
                {
                    "is_test": True,
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
            profit_outliers = format_dataframe(profit_outliers,
                                               {"leverage": ".4f", "dffits": ".4f", "resid_stu": ".4f", "cook": ".4f"})
            res.append(self.transform_table_data_to_html(
                {
                    "is_test": True,
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
            plt.xlabel("predict value")
            plt.ylabel('true value')
            res.append({
                "is_test": False,
                "title": "预测值与真实值对比散点图",
                "base64": "{}".format(self.plot_and_output_base64_png(plt))
            })
        return res

    # 聚类结果可视化
    def show_cluster_result(self, x, model):
        res = []
        if len(x.columns) == 2:
            x_new = x.values
        elif len(x.columns) > 2:
            try:
                from sklearn.decomposition import PCA
            except:
                raise NotImplementedError("cannot import sklearn PCA")
            pca = PCA(n_components=2).fit(x)
            x_new = pca.transform(x)
        else:
            raise ValueError("input feature's count must >= 2 ")

        x_with_label = pd.DataFrame(np.hstack((x_new, model.labels_.reshape(-1, 1))), columns=["0", "1", "2"])
        group_data = x_with_label.groupby(["2"])
        # 每个类绘制不同的颜色和marker
        color = ["r", "g", "b", "c", "k", "m", "y"]
        marker = ["+", "o", "*", ".", ",", "^", "1", "v", "<", ">",
                  "2", "3", "4", "s", "p", "h", "H", "D", "d", "|", "_"]
        legend_c = []
        legend_name = []
        for name, data in group_data:
            c = plt.scatter(data["0"], data["1"], c=random.sample(color, 1)[0], marker=random.sample(marker, 1)[0])
            legend_c.append(c)
            legend_name.append(str(int(name)))
        plt.legend(legend_c, legend_name)
        # 可视化结果转base64输出
        res.append({
            "is_test": False,
            "title": "聚类结果可视化",
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
                res.extend(self.show_cluster_result(x, model))
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

    # 执行插入sql数据
    @staticmethod
    def exec_insert_sql(table, key_list, value_list):
        conn = pymysql.connect(host='rm-2ze5vz4174qj2epm7so.mysql.rds.aliyuncs.com', port=3306, user='yzkj',
                               password='yzkj2020@', database='sophia_manager', charset='utf8')
        cursor = conn.cursor()
        sql_list = []
        for key in key_list:
            for value in value_list:
                sql = "INSERT INTO {}({}) VALUES ('{}')".format(table, ",".join(key), "','".join(value))
                sql_list.append(sql)
                try:
                    # Execute the SQL command
                    cursor.execute(sql)
                    # Commit your changes in the database
                    conn.commit()
                except Exception as e:
                    log.error(e)
                    # Rollback in case there is any error
                    conn.rollback()
        conn.close()
        return sql_list

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
