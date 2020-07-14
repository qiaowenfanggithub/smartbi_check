# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : flask_test

Description : 

Author : leiliang

Date : 2020/6/28 4:41 下午

--------------------------------------------------------

"""
import requests
import pandas as pd

if __name__ == '__main__':
    # data = pd.read_csv("./data/PimaIndiansdiabetes.csv")
    data = pd.read_excel("./data/buy-computer.xlsx")
    my_session = requests.session()
    # ======================= 单因素方差分析 =============================
    # 特征编码
    kwargs = {
        "table_name": "anova_one_way",  # str,数据库表名
        "X": ["group"],  # list,自变量
        "Y": ["value"],  # list,因变量
        "alpha": "0.05",  # str,置信区间百分比
    }
    res = my_session.post(url='http://127.0.0.1:9012/', json=kwargs, timeout=30)
    # res = my_session.post(url='http://121.42.242.214:8082', json=kwargs, timeout=5)
    # res = my_session.post(url='http://47.105.136.165:8082/', json=kwargs, timeout=30)
    # res = my_session.post(url='http://47.105.136.165:8085/index', json=kwargs, timeout=30)
    print(res.text)
