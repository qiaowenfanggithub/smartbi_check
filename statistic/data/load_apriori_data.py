# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : load_apriori_data

Description : 

Author : leiliang

Date : 2020/8/14 11:52 上午

--------------------------------------------------------

"""
import pandas as pd
dataset = [
    ['牛奶', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
    ['莳萝', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
    ['牛奶', '苹果', '芸豆', '鸡蛋'],
    ['牛奶', '独角兽', '玉米', '芸豆', '酸奶'],
    ['玉米', '洋葱', '洋葱', '芸豆', '冰淇淋', '鸡蛋']
]
# data = pd.DataFrame(dataset)
# print(data)
# data.to_csv("apriori_test.csv")

d = pd.read_csv("apriori_test.csv")
print(d)