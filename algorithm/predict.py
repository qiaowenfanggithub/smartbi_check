# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : evaluate

Description : 总的预测函数

Author : leiliang

Date : 2020/8/4 10:59 上午

--------------------------------------------------------

"""
import logging
from base_algorithm import BaseAlgorithm
from utils import transform_table_data_to_html, format_dataframe

log = logging.getLogger(__name__)


class predictModel(BaseAlgorithm):
    def __init__(self):
        BaseAlgorithm.__init__(self)
        self.get_config()
        if self.config["oneSample"]:
            self.table_data = None
        else:
            self.table_data = self.exec_sql(self.config['tableName'], self.config['X'], None)
        self.model = self.load_model_by_database(self.config['algorithm'], self.config['model'])

    def get_config(self):
        '''
            前端传过来的参数
        {
            "algorithm": "",
            "model": "",
            "oneSample": False,
            "table": "",
            "X": "",
        }
        :return:
        '''
        self.config = {}
        try:
            self.config['algorithm'] = self.web_data['algorithm']
            self.config['model'] = self.web_data['model']
            self.config['oneSample'] = self.web_data['oneSample']
            self.config['tableName'] = self.web_data.get('tableName', None)
            self.config['X'] = self.web_data.get('X')
        except Exception as e:
            log.info(e)
            raise e

    def model_predict(self):
        try:
            res = {}
            if self.config['oneSample']:
                if not self.config['X']:
                    raise ValueError("feature must not be empty when one-sample")
                X = [[float(x) for x in self.config['X']]]
                predict = self.model.predict(X)[0] if isinstance(self.model.predict(X)[0], str) else "{:.0f}".format(
                    self.model.predict(X)[0])
                res.update({
                    "data": [[",".join([str(s) for s in self.config['X']]), predict]],
                    "title": "单样本预测结果",
                    "col": ["样本特征", "模型预测结果"],
                })
            else:
                # 从数据库拿数据
                if not self.config['tableName'] or self.config['tableName'] == "":
                    raise ValueError("cannot find table data when predict many samples")
                data = self.table_data
                log.info("输入数据大小:{}".format(len(data)))
                data = data.astype(float)
                data["predict"] = self.model.predict(data.values)
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
            return {"data": "", "code": "500", "msg": "{}".format(e.args)}
