# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : data_preprocess

Description : 

Author : leiliang

Date : 2020/7/10 4:22 下午

--------------------------------------------------------

"""
from utils import *
import pandas as pd

if __name__ == '__main__':
    # 决策树数据处理
    decision_data_path = "buy-computer.csv"
    decision_data = pd.read_csv(decision_data_path)
    # 数据归一化
    decision_data = data_encoder(decision_data, decision_data.columns)
    print(decision_data)
    decision_data.to_csv(decision_data_path.split(".")[0] + "-new.csv", index=None)