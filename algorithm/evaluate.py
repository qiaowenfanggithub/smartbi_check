# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : evaluate

Description : 总的评估函数

Author : leiliang

Date : 2020/8/4 10:59 上午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm

log = logging.getLogger(__name__)


class evaluateModel(BaseAlgorithm):
    def __init__(self):
        BaseAlgorithm.__init__(self)
        self.get_config()
        self.table_data = self.exec_sql(self.config['tableName'], self.config['X'], self.config['Y'])
        self.model = self.load_model_by_database(self.config['algorithm'], self.config['model'])

    def get_config(self):
        '''
            前端传过来的参数
        {
            "algorithm": "",
            "model": "",
            "table": "",
            "X": "",
            "Y": "",
            "show_options": ["report", "matrix", "roc"]
        }
        :return:
        '''
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

    def model_evaluate(self):
        res = []
        try:
            x_test = self.table_data.loc[:, self.config['X']]
            y_test = self.table_data[self.config['Y'][0]]

            # 分类结果可视化
            if self.config['algorithm'] in ["logisticRegression", "decisionTree",
                                            "randomForest", "svmClassifier"]:
                res.extend(self.algorithm_show_result(self.model, x_test, y_test,
                                                      options=self.config['show_options'],
                                                      method="classifier"))

            # 回归结果可视化
            if self.config['algorithm'] in ["logisticRegression", "linerRegression",
                                            "polyLinerRegression"]:
                model_name = self.config["model"][:18] + "2" + self.config["model"][18:]
                self.model = self.load_model_by_database("logisticRegression2", model_name)
                res.extend(self.algorithm_show_result(self.model, x_test, y_test,
                                                      options=self.config['show_options'],
                                                      method="regression"))

            # 聚类结果可视化
            if self.config['algorithm'] in ["kMeans", "hierarchicalCluster"]:
                res.extend(self.algorithm_show_result(self.model, x_test, y_test,
                                                      options=self.config['show_options'],
                                                      method="cluster"))

            response_data = {"res": res,
                             "code": "200",
                             "msg": "ok!"}
            return response_data
        except Exception as e:
            # raise e
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}
