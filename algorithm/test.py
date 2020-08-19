# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/9 10:33 下午

--------------------------------------------------------

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 导入数据
wine = pd.read_csv('wine.csv', names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
                                      "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                                      "Color_intensity", "Hue", "OD280", "Proline"])
# print(wine.head())
X = wine.drop("Cultivator", axis=1)
Y = wine["Cultivator"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)  # 训练集、测试集

# 标准化
scaler = StandardScaler()
scaler.fit(X_train)  # 利用训练集计算标准化的参数
X_train = scaler.transform(X_train)  # 利用上面计算的参数标准化训练集和测试集
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, Y_train)  # 训练模型
Y_test_pred = mlp.predict(X_test)  # 预测
print("预测结果是：", Y_test_pred)
print("预测正确率为：", np.mean(Y_test_pred == Y_test))
