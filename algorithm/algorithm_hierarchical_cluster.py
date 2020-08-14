# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : algorithm_random_forest

Description : 层级聚类算法封装

Author : leiliang

Date : 2020/7/30 11:15 下午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from sklearn.cluster import AgglomerativeClustering
from utils import transform_table_data_to_html, format_dataframe

log = logging.getLogger(__name__)


class hierarchicalCluster(BaseAlgorithm):
    def __init__(self, method):
        BaseAlgorithm.__init__(self)
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
            self.table_data = self.exec_sql(self.config['tableName'], self.config['X'])

    def get_train_config_from_web(self):
        """
        接口请求参数:{
            "tableName": "", # str,数据库表名
            "X": ["x1", "x2"], # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
            "param":{
                "n_clusters": "3", # str,聚类的个数
            }
        :return:
        """
        self.config = {}
        try:
            self.config['tableName'] = self.web_data['tableName']
            self.config['X'] = self.web_data.get('X')
            self.config['n_clusters'] = int(self.web_data['param']['n_clusters'])
            self.config['linkage'] = self.web_data['param']['linkage']
            self.config['affinity'] = self.web_data['param']['affinity']
            self.config['show_options'] = self.web_data.get("show_options", [])
        except Exception as e:
            log.info(e)
            raise e

    def get_evaluate_config_from_web(self):
        pass

    def get_predict_config_from_web(self):
        self.config = {}
        try:
            self.config['algorithm'] = self.web_data['algorithm']
            self.config['model'] = self.web_data['model']
            self.config['oneSample'] = self.web_data.get('oneSample', False)
            self.config['tableName'] = self.web_data.get('tableName')
            self.config['X'] = self.web_data.get('X')
        except Exception as e:
            log.info(e)
            raise e

    def train(self):
        try:
            # 模型训练和网格搜索
            self.table_data = self.table_data.astype("float")
            hie_cluster = AgglomerativeClustering(n_clusters=self.config["n_clusters"],
                                                  affinity=self.config["affinity"],
                                                  linkage=self.config["linkage"])
            self.model = hie_cluster.fit(self.table_data)

            # 保存模型
            # self.save_model(self.model, "kMeans")
            model_info = self.save_model_into_database("hierarchicalCluster")

            # 聚类结果可视化
            res = self.algorithm_show_result(self.model, self.table_data, None,
                                             options=self.config['show_options'],
                                             method="cluster")

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
        pass

    def predict(self):
        try:
            # model = self.load_model("kMeans")
            model = self.load_model_by_database(self.config["algorithm"], self.config["model"])
            res = {}
            if self.config['oneSample']:
                if not self.config['X']:
                    raise ValueError("feature must not be empty when one-sample")
                X = [[float(x) for x in self.config['X']]]
                predict = model.fit_predict(X)[0] if isinstance(model.fit_predict(X)[0], str) else "{:.0f}".format(
                    model.fit_predict(X)[0])
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
                data["predict"] = model.fit_predict(data.values)
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
        return "hierarchicalCluster"
