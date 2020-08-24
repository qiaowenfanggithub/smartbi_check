# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : gen_dataset

Description : 生成机器学习训练集，用于统计分析平台测试

Author : leiliang

Date : 2020/8/21 1:42 下午
--------------------------------------------------------

"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, \
    load_diabetes, load_boston, load_linnerud, load_wine


def load_dataset(dataset, file_name):
    try:
        data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        label = pd.DataFrame(dataset.target, columns=['label'])
        # data_with_label = pd.concat([data, label], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,
                                                            random_state=random_state, shuffle=is_shuffle)

        train_data = pd.concat([x_train, y_train], axis=1).reset_index()
        train_data.to_csv("{}_train.csv".format(file_name))
        test_data = pd.concat([x_test, y_test], axis=1).reset_index()
        test_data.to_csv("{}_test.csv".format(file_name))
        return "success"
    except Exception as e:
        raise e


def gen_office_dataset(dataset_name, is_shuffle=True, random_state=2020):
    """
    生成sklearn官方数据集
    :param dataset_name: 数据集名称
    :return:
    """
    if dataset_name == "iris":
        dataset = load_iris()
    elif dataset_name == "":
        pass
    else:
        pass
    res = load_dataset(dataset, dataset_name)
    print(res)


def make_dataset():
    pass


if __name__ == '__main__':
    gen_office_dataset("iris")
