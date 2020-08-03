# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : load_iris

Description : 

Author : leiliang

Date : 2020/8/3 4:25 下午

--------------------------------------------------------

"""
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

iris_data = pd.DataFrame(iris["data"], columns=["sepal length", "sepal width", "petal length", "petal width"])
iris_data.to_csv("./data/iris_kmeans.csv")