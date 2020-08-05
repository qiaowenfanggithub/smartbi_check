# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : load_iris

Description : 

Author : leiliang

Date : 2020/8/3 4:25 下午

--------------------------------------------------------

"""
import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from base64_to_png import base64_to_img
from utils import plot_and_output_base64_png


iris = load_iris()
# print(iris)

iris_data = pd.DataFrame(iris["data"], columns=["sepal length", "sepal width", "petal length", "petal width"])
# iris_data.to_csv("./data/iris_kmeans.csv")

iris_data_2 = iris_data[["sepal length", "sepal width"]]
# iris_data_2.to_csv("./data/iris_kmeans_2.csv")

x_with_label = pd.DataFrame(np.hstack((iris_data_2, iris["target"].reshape(-1, 1))), columns=["0", "1", "2"])
group_data = x_with_label.groupby(["2"])

color = ["r", "g", "b", "c", "k", "m", "w", "y"]
marker = ["+", "o", "*", ".", ",", "^", "1", "v", "<", ">", "2", "3", "4", "s", "p", "h", "H", "D", "d", "|", ""]
legend_c = []
legend_name = []
for name, data in group_data:
    c = plt.scatter(data["0"], data["1"], c=np.random.choice(color, 1, replace=False)[0], marker=np.random.choice(marker, 1, replace=False)[0])
    legend_c.append(c)
    legend_name.append(str(int(name)))
plt.legend(legend_c, legend_name)
plt.show()