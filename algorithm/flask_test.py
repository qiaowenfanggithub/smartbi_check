# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : flask_test

Description : 

Author : leiliang

Date : 2020/6/28 4:41 下午

--------------------------------------------------------

"""
from __future__ import print_function
import requests

if __name__ == '__main__':
    my_session = requests.session()
    # ======================= 决策树-训练 =============================
    # kwargs = {
    #     "isTrain": False,  # str,数据库表名
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,特征
    #     "Y": ["是否购买电脑"],  # list,标签
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "2",  # str,几折交叉验证
    #     "param": {
    #         "criterion": ["gini"],  # 不纯度指标gini、entropy
    #         "max_depth": ["2"],  # 指定树的最大深度
    #         "min_samples_split": ["2"],  # :int, float, optional (default=2)。表示分裂一个内部节点需要的最少样本数。
    #         "min_samples_leaf": ["1"],  # int, float, optional (default=1)。指定每个叶子节点需要的最少样本数。
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/train', json=kwargs, timeout=30)

    # ======================= 决策树-预测(多个测试样本) =============================
    # kwargs = {
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/predict', json=kwargs, timeout=30)

    # ======================= 决策树-预测(单个测试样本) =============================
    # kwargs = {
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": [0, 0, 0, 0],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/predict', json=kwargs, timeout=30)

    # ======================= 逻辑回归-训练 =============================
    # kwargs = {
    #     "isTrain": True,  # True,进行训练还是测试
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "3",  # str,几折交叉验证
    #     "param": {
    #         "penalty": ["l2"],  # str,惩罚项
    #         "C": ["2"],  # str,惩罚项系数
    #         "solver": ["saga"],  # str，优化算法
    #         "max_iter": ["1000"],  # str，最大迭代步数
    #     },
    #     "show_options": ["matrix", "roc", "r2", "coff"]
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/logistics/train', json=kwargs, timeout=30)

    # ======================= 逻辑回归-预测(多个测试样本) =============================
    kwargs = {
        "oneSample": False,  # 是否批量上传数据进行预测
        "tableName": "buy_computer_new",  # str,数据库表名
        # "X": [1,1,1,0],  # list,自变量
        "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    }
    res = my_session.post(url='http://127.0.0.1:5000/algorithm/logistics/predict', json=kwargs, timeout=30)

    # ======================= 逻辑回归-预测(单个测试样本) =============================
    # kwargs = {
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": [1, 0, 0, 0],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/logistics/predict', json=kwargs, timeout=30)

    # ======================= K-Means 聚类 =============================
    # kwargs = {
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，每个元素是浮点类型
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "param": {
    #         "n_clusters": ["2", "3"],  # list,聚类中心数量，默认2个,如果是多个画图展示每个聚类的效果
    #         "max_iter": "1000",  # str，最大迭代步数，默认1000个
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/kMeans', json=kwargs, timeout=30)

    # ======================= 随机森林-训练 =============================
    # kwargs = {
    #     "isTrain": False,  # True,进行训练还是测试
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量，当表格方向为h时表示多个变量名，为v时表示分类变量字段
    #     "Y": ["是否购买电脑"],  # list,因变量,当表格方向为v是使用
    #     "rate": "0.3",  # str,测试集训练集分割比例
    #     "randomState": "2020",  # str,测试集训练集分割比例时的随机种子数
    #     "cv": "3",  # str,几折交叉验证
    #     "param": {
    #         "n_estimators": [100],  # list,树的个数
    #         "criterion": ["gini"],  # list,树划分准则
    #         "max_features": [2],  # list，用于训练的最大特征数量
    #         "max_depth": [5],  # list，树的最大深度列表
    #         "min_samples_split": [2],  # list， 内部节点再划分所需最小样本数
    #         "min_samples_leaf": [1],  # list，叶子节点最少样本数
    #     }
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/train', json=kwargs, timeout=30)

    # ======================= 决策树-预测(多个测试样本) =============================
    # kwargs = {
    #     "oneSample": False,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": ["年龄", "收入层次", "是否单身", "信用等级"],  # list,自变量
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/randomForest/predict', json=kwargs, timeout=30)

    # ======================= 决策树-预测(单个测试样本) =============================
    # kwargs = {
    #     "oneSample": True,  # 是否批量上传数据进行预测
    #     "tableName": "buy_computer_new",  # str,数据库表名
    #     "X": [0, 0, 0, 0],  # list,自变量，每个元素是浮点类型
    # }
    # res = my_session.post(url='http://127.0.0.1:5000/algorithm/decisionTree/predict', json=kwargs, timeout=30)
    print(res.text)
