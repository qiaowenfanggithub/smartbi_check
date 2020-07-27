# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm_logistic

Description : 

Author : leiliang

Date : 2020/7/27 10:41 上午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


class logisticAlgorithm(BaseAlgorithm):
    def __init__(self, method):
        BaseAlgorithm.__init__(self)
        # super(logisticAlgorithm, self).__init__()
        if method == "train":
            self.get_train_config_from_web()
        elif method == "evaluate":
            self.get_evaluate_config_from_web()
        else:
            self.get_predict_config_from_web()
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
            "rate": "0.3", # str,测试集训练集分割比例
            "randomState": "2020", # str,测试集训练集分割比例时的随机种子数
            "cv": "10", # str,几折交叉验证
            "param":{
                "penalty": "l2", # str,惩罚项
                "C": "2", # str,惩罚项系数
                "solver": "saga", # str，优化算法
                "max_ter": "1000", # str，最大迭代步数
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
            self.config['randomState'] = int(self.web_data.get('randomState', 0))
            self.config['rate'] = float(self.web_data.get('rate', 0))
            self.config['cv'] = int(self.web_data.get('cv', 0))
            self.config['show_options'] = self.web_data.get("show_options", [])
            self.config['param']["penalty"] = self.config['param'].get("penalty", "")
            self.config['param']["C"] = self.config['param'].get("C", ["0"])
            self.config['param']["C"] = [float(c) for c in self.config['param']["C"]]
            # 默认saga随机梯度下降
            self.config['param']["solver"] = self.config['param'].get("solver", ["saga"])
            self.config['param']["max_iter"] = self.config['param'].get("max_iter", ["1000"])
            self.config['param']["max_iter"] = [int(m) for m in self.config['param']["max_iter"]]
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
            self.config['oneSample'] = self.web_data['oneSample']
            self.config['tableName'] = self.web_data.get('tableName')
            self.config['X'] = self.web_data.get('X')
            self.config['Y'] = self.web_data.get('Y')
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.info(e)
            raise e

    def train(self):
        try:
            # 划分测试集和训练集
            x_train, x_test, y_train, y_test = self.split_data(self.table_data, self.config)

            # 模型训练和网格搜索
            clf = LogisticRegression(random_state=self.config["randomState"])
            self.model = GridSearchCV(clf, self.config['param'], cv=self.config['cv'])
            self.model.fit(x_train, y_train)
            best_param = self.model.best_params_
            self.model = LogisticRegression(**best_param, random_state=self.config["randomState"]).fit(x_test, y_test)

            # 保存模型
            self.save_model(self.model, "logisticRegression")

            # 结果可视化
            res = self.algorithm_show_result(self.model, x_test, y_test,
                                             options=self.config['show_options'],
                                             method=["classifier", "regression"])

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!",
                             }
            return response_data
        except Exception as e:
            return {"data": "", "code": "500", "msg": "".format(e.args)}

    def evaluate(self):
        try:
            model = self.load_model("logisticRegression")
            x_test = self.table_data.loc[:, self.config['X']]
            y_test = self.table_data[self.config['Y'][0]]

            res = self.algorithm_show_result(model, x_test, y_test,
                                             options=self.config['show_options'],
                                             method=["classifier", "regression"])

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            return {"data": "", "code": "500", "msg": "".format(e.args)}

    def predict(self):
        try:
            model = self.load_model("logisticRegression")
            res = {}
            if self.config['oneSample']:
                if not self.config['X']:
                    raise ValueError("feature must not be empty when one-sample")
                X = [float(x) for x in self.config['X']]
                res.update({"predict": model.predict([X]),
                            "title": "单样本预测结果"})
            else:
                # 从数据库拿数据
                if not self.config['tableName']:
                    raise ValueError("cannot find table data when multi-sample")
                data = self.exec_sql(self.config['tableName'], self.config['X'])
                log.info("输入数据大小:{}".format(len(data)))
                data = data.astype(float)
                pre_data = model.predict(data.values)
                res.update({
                    "predict": pre_data,
                    "title": "多样本预测结果",
                    "col": "预测结果"
                })
            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            return {"data": "", "code": "500", "msg": "".format(e.args)}
