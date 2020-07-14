# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/2 3:03 下午

--------------------------------------------------------

"""
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm

iris = datasets.load_iris()
parameters = {'kernel': ('linear', "rbf"), 'C': [1, 10]}  # 注意score='roc_auc'是二分类的,多分类会报错
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=7)
clf.fit(iris.data, iris.target)
print(type(clf.best_params_))
print(clf.best_params_)
