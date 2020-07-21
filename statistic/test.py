# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/21 12:09 下午

--------------------------------------------------------

"""
import pandas as pd

a = pd.DataFrame([[1.000001, 2, 3], [4, 5.00003, 6.111111115]], columns=["A", "B", "C"])
print(a)
b = a.round({"A": 2, "B": 2})
print(b)

