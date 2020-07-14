# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : decision_tree

Description : 

Author : leiliang

Date : 2020/7/2 2:26 下午

--------------------------------------------------------

"""
from flask import request, Flask, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import logging
from utils import get_data_from_mysql

log = logging.getLogger(__name__)
log.info('popularity_predict_init...')

app = Flask(__name__)


@app.route('/train', methods=['GET', 'POST'])
def model_train():
    """
    接口请求参数:{
        "table_name": "" # str,数据库表名
        "features": ["x1", "x2"], # list
        "label": ["y"], # list
        "train_test_split": "0.3" # str, default=0.3
        "grid_search_cv": False # bool, default=False
        "param":{
            "criterion": "gini", # str, 划分节点的指标(gini, entropy), default="gini"
            "splitter": "best", # str, 待研究
            "max_depth": None, # str, default=None
            "min_sample_split": None, # str, default=None,
            "min_sample_leaf": None, # str, default=None
        }
    }
    :return:接口返回参数{
        "": "",
    }
    """
    log_file = "decision_tree.log"
    logging.basicConfig(filename=log_file,
                        format="%(asctime)s [ %(levelname)-6s ] %(message)s",
                        level='INFO')
    logging.getLogger().addFilter(logging.StreamHandler())
    logging.getLogger("requests").setLevel(logging.WARNING)
    try:
        request_data = request.json
        log.info(request_data)
    except Exception as e:
        log.info(e)
        raise e
    log.info("receive json request is:{}".format(request_data))
    try:
        table_name = request_data['table_name']
        features = request_data['features']
        lable = request_data['lable']
        param = request_data["param"]
        train_test_split = request_data.get('train_test_split', 0.3)
        grid_search_cv = request_data.get('grid_search_cv', False)
    except Exception as e:
        log.info(e)
        return jsonify({"code": "500", "msg": "input config error:{}".format(e.args)})

    # 根据数据库表名取数据
    sql_sentence = "select {} from {}".format(",".join(features + lable), table_name)
    data = get_data_from_mysql(sql_sentence)
    return data

    # # 判断是否上传成功
    # data = None
    # len = data.shape[1] - 1
    # data_X = data[:, 0:len]
    # data_y = data[:, len]
    # # 数据分割
    # x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, random_state=666)
    #
    # model = DecisionTreeClassifier()
    # model.fit(x_train, y_train)
    #
    # # 模型评估
    # print(model.score(x_train, y_train))
    # print(model.score(x_test, y_test))
    #
    # # 模型的保存与持久化
    # joblib.dump(model, "classifier_decisionTree.model")  # 将训练后的线性模型保存
    #
    # return '模型训练完成！'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file_model = request.files['model']

    file_predict = request.files['predict']
    # 读取csv文件
    predict_data = np.loadtxt(file_predict, delimiter=",")
    predict_data1 = predict_data[:, :]

    print(predict_data)

    model = joblib.load(file_model)
    result = model.predict(predict_data1)

    return " ".join('%s' % id for id in result)


# 检查文件后缀名是否是文本文件
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True, port=9012)
