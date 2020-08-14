# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm_poly_regression

Description :

Author : qiaowenfang

Date : 2020/8/3 11:36 上午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from utils import transform_table_data_to_html, gen_poly_col
import statsmodels.api as sm

log = logging.getLogger(__name__)


class polyRegression(BaseAlgorithm):
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
            "M": [{'x1':2,'x2':3}] # 自变量相对应的用户指定的最高阶数
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
            self.config['M'] = self.web_data.get('M')
            self.config['param'] = self.web_data['param']
            self.config['param']["fit_intercept"] = self.config['param'].get("fit_intercept", True)
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.exception("Exception Logged")
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
            log.exception("Exception Logged")
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
            log.exception("Exception Logged")
            raise e

    def train(self):
        try:
            # 划分测试集和训练集
            # x_train, x_test, y_train, y_test = self.split_data(self.table_data, self.config)
            self.table_data = self.table_data.astype(float)
            data = gen_poly_col(self.table_data, self.config['M'][0])
            data['newy'] = data[self.config["Y"][0]]
            del data[self.config["Y"][0]]
            x_train = data.iloc[:, :-1]
            y_train = data.iloc[:, -1]

            # 模型训练
            x_train = x_train.astype(float)
            y_train = y_train.astype(float)
            if self.config["param"]["fit_intercept"]:
                x = sm.add_constant(x_train)
                self.model = sm.OLS(y_train, x).fit()
            else:
                self.model = sm.OLS(y_train, x_train).fit()

            # 保存模型
            # self.save_model(self.model, "polyLinerRegression")
            model_info = self.save_model_into_database("polyLinerRegression")

            # 结果可视化
            x_train = x_train.astype(float)
            y_train = y_train.astype(float)
            res = self.algorithm_show_result(self.model, x_train, y_train,
                                             options=self.config['show_options'],
                                             method="regression")

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
            log.exception("Exception Logged")
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    def predict(self):
        try:
            try:
                import statsmodels.api as sm
            except:
                raise ImportError("statsmodels.api cannot import")
            # model = self.load_model("linerRegression")
            self.model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            res = {}
            if self.config['oneSample']:
                if len(self.config['X']) == 0 or self.config['X'][0] == "":
                    raise ValueError("feature must not be empty when one-sample")
                if "const" in self.model.params:
                    X = [1.] + [float(x) for x in self.config['X']]
                else:
                    X = [float(x) for x in self.config['X']]
                X = self.get_poly_data_from_model_params(X)
                res.update({
                    "data": [[",".join(self.model.params.index[1:]), "{:.4f}".format(self.model.predict(X)[0])]],
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
                if "const" in self.model.params:
                    data = sm.add_constant(data)
                data = self.get_poly_data_from_model_params(data)
                data["predict"] = self.model.predict(data)
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
            log.exception("Exception Logged")
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}

    # 预测的时候从模型拿到每个自变量对于的最高阶数
    def get_poly_data_from_model_params(self, data):
        for x in self.model.params.index:
            if "**" not in x:
                continue
            if isinstance(data, list):
                x_list = [c for c in self.model.params.index if c != "const" and "**" not in c]
                idx = x_list.index(x.split("**")[0])
                i = int(x.split("**")[1])
                data.append(self.config["X"][idx] ** i)
            else:
                data[x] = data[x.split("**")[0]] ** int(x.split("**")[1])
        return data

    def __str__(self):
        return "polyRegression"
