# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm_liner_regression

Description : 

Author : leiliang

Date : 2020/7/27 9:55 下午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from utils import transform_table_data_to_html
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class linerRegression(BaseAlgorithm):
    def __init__(self, method):
        BaseAlgorithm.__init__(self)
        # super(logisticAlgorithm, self).__init__()
        if method == "train":
            self.get_train_config_from_web()
        elif method == "evaluate":
            self.get_evaluate_config_from_web()
        elif method == "predict":
            self.get_predict_config_from_web()
        elif method == "visualization":
            self.get_visualization_config_from_web()
        else:
            raise ValueError("input method:{} is not supported".format(method))
        if method == 'predict' and self.config["oneSample"]:
            self.table_data = None
        else:
            self.table_data = self.exec_sql(self.config['tableName'], self.config['X'], self.config['Y'])

    def get_train_config_from_web(self):
        """
        接口请求参数:{
            "tableName": "", # str,数据库表名
            "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "Y": ["y"], # list,因变量,当表格方向为v是使用
            "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
            "rate": "0.3", # str,测试集训练集分割比例
            "param":{"fit_intercept": True}, # bool,True或者False，是否有截距项
            "show_options": ["r2", "coff", "Independence"]
            }
        :return:
        """
        self.config = {}
        try:
            self.config['tableName'] = self.web_data['tableName']
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['param'] = self.web_data['param']
            self.config['param']["fit_intercept"] = self.config['param'].get("fit_intercept", True)
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.info(e)
            raise e

    def get_evaluate_config_from_web(self):
        """
        接口请求参数:{
            "tableName": "", # str,数据库表名
            "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "Y": ["y"], # list,因变量,当表格方向为v是使用
            "show_options": ["matrix", "roc", "r2", "coff"]
            }
        :return:
        """
        self.config = {}
        try:
            self.config['algorithm'] = self.web_data['algorithm']
            self.config['model'] = self.web_data['model']
            self.config['tableName'] = self.web_data['tableName']
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.info(e)
            raise e

    def get_predict_config_from_web(self):
        """
        接口请求参数:{
            "oneSample": False,  # 是否批量上传数据进行预测
            "tableName": "buy_computer_new",  # str,数据库表名
            # "X": [1,1,1,0],  # list,自变量，当单样本时是一个向量
            "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，多个样本情况
        :return:
        """
        self.config = {}
        try:
            self.config['algorithm'] = self.web_data['algorithm']
            self.config['model'] = self.web_data['model']
            self.config['oneSample'] = self.web_data['oneSample']
            self.config['tableName'] = self.web_data.get('tableName')
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.info(e)
            raise e

    def get_visualization_config_from_web(self):
        """
        接口请求参数
            "tableName": "advertising",  # str,数据库表名
            "X": ["TV", "radio", "newspaper"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "Y": ["sales"],  # list,因变量,当表格方向为v是使用
            "show_options": ["y_count", "pairs", "corr", "y_corr"], # 展示选项
            "x_count": [], # list，选择要展示频率分布直方图的自变量
            "box": [], # list，选择要展示箱型图的自变量
        :return:
        """
        self.config = {}
        try:
            self.config['tableName'] = self.web_data.get('tableName')
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['show_options'] = self.web_data.get("show_options", [])
            self.config['x_count'] = self.web_data.get("x_count", [])
            self.config['box'] = self.web_data.get("box", [])
        except Exception as e:
            log.info(e)
            raise e

    def train(self):
        try:
            # 划分测试集和训练集
            # x_train, x_test, y_train, y_test = self.split_data(self.table_data, self.config)
            x_train = self.table_data[self.config["X"]]
            y_train = self.table_data[self.config["Y"][0]]

            # 模型训练
            x_train = x_train.astype(float)
            y_train = y_train.astype(float)
            if self.config["param"]["fit_intercept"]:
                x = sm.add_constant(x_train)
                self.model = sm.OLS(y_train, x).fit()
            else:
                self.model = sm.OLS(y_train, x_train).fit()

            # 保存模型
            # self.save_model(self.model, "linerRegression")
            self.save_model_into_database("linerRegression")

            # 结果可视化
            x_train = x_train.astype(float)
            y_train = y_train.astype(float)
            res = self.algorithm_show_result(self.model, x_train, y_train,
                                             options=self.config['show_options'],
                                             method="regression")

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!",
                             }
            return response_data
        except Exception as e:
            # raise e
            log.error(e)
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def evaluate(self):
        try:
            # model = self.load_model("linerRegression")
            model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            x_test = self.table_data.loc[:, self.config['X']]
            y_test = self.table_data[self.config['Y'][0]]

            res = self.algorithm_show_result(model, x_test, y_test,
                                             options=self.config['show_options'],
                                             method="regression")

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            log.error(e)
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def predict(self):
        try:
            try:
                import statsmodels.api as sm
            except:
                raise ImportError("statsmodels.api cannot import")
            # model = self.load_model("linerRegression")
            model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            res = {}
            if self.config['oneSample']:
                if len(self.config['X']) == 0 or self.config['X'][0] == "":
                    raise ValueError("feature must not be empty when one-sample")
                if "const" in model.params:
                    X = [1.] + [float(x) for x in self.config['X']]
                else:
                    X = [float(x) for x in self.config['X']]
                res.update({
                    "data": [[",".join([str(s) for s in self.config['X']]), "{:.4f}".format(model.predict(X)[0])]],
                    "title": "单样本预测结果",
                    "col": ["样本特征", "模型预测结果"],
                })
            else:
                # 从数据库拿数据
                if not self.config['tableName']:
                    raise ValueError("cannot find table data when multi-sample")
                data = self.exec_sql(self.config['tableName'], self.config['X'])
                log.info("输入数据大小:{}".format(len(data)))
                data = data.astype(float)
                if "const" in model.params:
                    data = sm.add_constant(data)
                data["predict"] = model.predict(data)
                data.drop(["const"], axis=1, inplace=True)
                res.update(transform_table_data_to_html({
                    "data": data.values.tolist(),
                    "title": "多样本预测结果",
                    "col": data.columns.tolist(),
                    "row": data.index.tolist()
                }))
            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            log.error(e)
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def visualization(self):
        """
        接口请求参数
            "tableName": "advertising",  # str,数据库表名
            "X": ["TV", "radio", "newspaper"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "Y": ["sales"],  # list,因变量,当表格方向为v是使用
            "show_options": ["y_count", "pairs", "corr", "y_corr"], # 展示选项
            "x_count": [], # list，选择要展示频率分布直方图的自变量
            "box": [], # list，选择要展示箱型图的自变量
        :return:
        """
        try:
            res = []
            self.table_data = self.table_data.astype("float")
            data = self.table_data.describe()
            res.append(transform_table_data_to_html({
                "data": data.values.tolist(),
                "title": "描述性统计分析",
                "col": data.columns.tolist(),
                "row": data.index.tolist()
            }))
            if self.config.get("x_count") and self.config.get("x_count")[0]:
                for x in self.config["x_count"]:
                    sns.distplot(self.table_data[x], kde=False)
                    # 显示纵轴标签
                    plt.ylabel("frequency")
                    # 显示图标题
                    plt.title("{} - frequency distribution histogram".format(x))
                    res.append({
                        "title": "{} 分布直方图".format(x),
                        "base64": "{}".format(self.plot_and_output_base64_png(plt))
                    })
            if "y_count" in self.config["show_options"]:
                sns.distplot(self.table_data[self.config["Y"][0]], kde=False)
                # 显示横轴标签
                plt.xlabel("section")
                # 显示纵轴标签
                plt.ylabel("frequency")
                # 显示图标题
                plt.title("y frequency distribution histogram")
                res.append({
                    "title": "{} 分布直方图".format(self.config["Y"][0]),
                    "base64": "{}".format(self.plot_and_output_base64_png(plt))
                })
            if self.config.get("box") and self.config.get("box")[0]:
                for x in self.config["box"]:
                    sns.boxplot(self.table_data[x], palette="Set2", orient="v")
                    # 显示图标题
                    plt.title("{} - Box distribution to check outliers".format(x))
                    res.append({
                        "title": "{} 箱型图".format(x),
                        "base64": "{}".format(self.plot_and_output_base64_png(plt))
                    })
            if "pairs" in self.config["show_options"]:
                sns.pairplot(self.table_data)
                plt.title("Variable relation in pairs")
                res.append({
                    "title": "变量两两关系图",
                    "base64": "{}".format(self.plot_and_output_base64_png(plt))
                })
            if "corr" in self.config["show_options"]:
                corr = self.table_data.corr()
                sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                            linewidths=0.2, cmap="YlGnBu", annot=True)
                plt.title("Correlation between variables")
                res.append({
                    "title": "相关系数图",
                    "base64": "{}".format(self.plot_and_output_base64_png(plt))
                })
            if "y_corr" in self.config["show_options"]:
                self.table_data.corr()[self.config["Y"][0]].sort_values(ascending=False).plot(kind='bar')
                plt.title("Correlations between y and x")
                res.append({
                    "title": "因变量和各自变量的相关系数图",
                    "base64": "{}".format(self.plot_and_output_base64_png(plt))
                })
            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def __str__(self):
        return "liner_regression"
