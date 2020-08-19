# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm_mlp

Description : 多层感知机分类实现

Author : leiliang

Date : 2020/7/30 11:15 下午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from utils import transform_table_data_to_html, format_dataframe

log = logging.getLogger(__name__)


class mlpClassifier(BaseAlgorithm):
    def __init__(self, method):
        BaseAlgorithm.__init__(self)
        self.one_type = "Classifier"
        self.one_type_name = "分类"
        self.second_type = "mlp"
        self.second_type_name = "多层感知机"
        # super(logisticAlgorithm, self).__init__()
        if method == "train":
            self.get_train_config_from_web()
        elif method == "evaluate":
            self.get_evaluate_config_from_web()
        else:
            self.get_predict_config_from_web()
        if method == 'predict':
            if self.config["oneSample"]:
                self.table_data = None
            else:
                self.table_data = self.exec_sql(self.config['tableName'], self.config['X'])
        else:
            self.table_data = self.exec_sql(self.config['tableName'], self.config['X'], self.config['Y'])

    def get_train_config_from_web(self):
        """
        接口请求参数:{
            "tableName": "", # str,数据库表名
            "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "Y": ["y"], # list,因变量,当表格方向为v是使用
            "rate": "0.3", # str,测试集训练集分割比例
            "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
            "cv": "10", # str,几折交叉验证
            "param":{
                "hidden_layer_sizes": "(10, 10)", # str（tuple）,隐藏层个数和每个隐藏层节点数
                "activation": "relu", # str,激活函数["identity", "logistic", "tanh", "relu"]
                "solver": "adam", # str，优化算法["lbfgs", "sgd", "adam"]
                "alpha": "0.0001", # str(float)，惩罚项系数["0.0001", "0.00001"]
                "batch_size": "auto", # str(int)，随机优化的minibatches的大小，默认auto，手动输入整数
                "learning_rate_int": "0.001", # str(float)，初始学习率
                "tol": "0.0001", # str(float)优化的容忍度
                "max_iter": "200", # str(int)最大迭代次数
            }
            "show_options": ["matrix", "roc", "r2", "coff"]
        :return:
        """
        self.config = {}
        try:
            self.config['tableName'] = self.web_data['tableName']
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['param'] = self.web_data['param']
            self.config['randomState'] = int(self.web_data.get('randomState', 666))
            self.config['rate'] = float(self.web_data.get('rate', 0.3))
            self.config['show_options'] = self.web_data.get("show_options", [])
            self.config["param"]["hidden_layer_sizes"] = (100, ) if not self.config["param"]["hidden_layer_sizes"] else tuple([int(c) for c in self.config["param"]["hidden_layer_sizes"]])
            self.config["param"]["activation"] = "relu" if not self.config["param"]["activation"] else self.config["param"]["activation"]
            self.config["param"]["solver"] = "adam" if not self.config["param"]["solver"] else self.config["param"]["solver"]
            self.config["param"]["alpha"] = 0.0001 if not self.config["param"]["alpha"] else float(self.config["param"]["alpha"])
            self.config["param"]["batch_size"] = "auto" if not self.config["param"]["batch_size"] or self.config["param"]["batch_size"] == "auto" else int(self.config["param"]["batch_size"])
            self.config["param"]["learning_rate_init"] = 0.001 if not self.config["param"]["learning_rate_init"] else float(self.config["param"]["learning_rate_init"])
            self.config["param"]["tol"] = 0.0001 if not self.config["param"]["tol"] else float(self.config["param"]["tol"])
            self.config["param"]["max_iter"] = 200 if not self.config["param"]["max_iter"] else int(self.config["param"]["max_iter"])
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
        except Exception as e:
            log.info(e)
            raise e

    def train(self):
        try:
            # 划分测试集和训练集
            x_train, x_test, y_train, y_test = self.split_data(self.table_data, self.config)

            # 模型训练和网格搜索
            clf = MLPClassifier(**self.config["param"], random_state=self.config["randomState"])
            self.model = clf.fit(x_train, y_train)

            # 保存模型
            # self.save_model(self.model, "randomForest")
            model_info = self.save_model_into_database("mlpClassifier")

            # 分类结果可视化
            res = self.algorithm_show_result(self.model, x_test, y_test,
                                             options=self.config['show_options'],
                                             method="classifier")

            response_data = {"res": res,
                             "model_info": model_info,
                             "code": "200",
                             "msg": "ok!",
                             }
            return response_data
        except Exception as e:
            # raise e
            log.exception("Exception Logged")
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def evaluate(self):
        res = []
        try:
            # model = self.load_model("randomForest")
            model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            x_test = self.table_data.loc[:, self.config['X']]
            y_test = self.table_data[self.config['Y'][0]]

            # 分类结果可视化
            res.extend(self.algorithm_show_result(model, x_test, y_test,
                                                  options=self.config['show_options'],
                                                  method="classifier"))

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            # raise e
            log.exception("Exception Logged")
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def predict(self):
        try:
            # model = self.load_model("randomForest")
            model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            res = {}
            if self.config['oneSample']:
                if not self.config['X']:
                    raise ValueError("feature must not be empty when one-sample")
                X = [[float(x) for x in self.config['X']]]
                predict = model.predict(X)[0] if isinstance(model.predict(X)[0], str) else "{:.0f}".format(model.predict(X)[0])
                res.update({
                    "data": [[",".join([str(s) for s in self.config['X']]), predict]],
                    "title": "单样本预测结果",
                    "col": ["样本特征", "模型预测结果"],
                })
            else:
                # 从数据库拿数据
                if not self.config['tableName'] or self.config['tableName'] == "":
                    raise ValueError("cannot find table data when multi-sample")
                data = self.table_data
                log.info("输入数据大小:{}".format(len(data)))
                data = data.astype(float)
                data["predict"] = model.predict(data.values)
                if data["predict"].dtypes != "object":
                    data = format_dataframe(data, {"predict": ".0f"})
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
            # raise e
            log.exception("Exception Logged")
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def __str__(self):
        return "random_forest"
