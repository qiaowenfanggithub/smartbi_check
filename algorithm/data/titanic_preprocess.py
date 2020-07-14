# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : titanic_preprocess

Description : 

Author : leiliang

Date : 2020/7/2 4:08 下午

--------------------------------------------------------

"""
# 导入数据
import pandas as pd

data = pd.read_csv('titanic.csv')
# print(data.head())
# print("==================================")
# 特征选择
data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)

# 处理缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())

# 删除含有空值的记录
data = data.dropna(axis=0)

# 男性为1（True），女性为0(False)
data['Sex'] = (data['Sex'] == 'male').astype('int')

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print(data.head())

# 重新保存
data.to_csv("tatinic_new.csv")
