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

a = pd.DataFrame({"A": [1, 2, 3, 3, 2], "B": ["a", "b", "c", "d", "e"],
                  "C": ["aa", "bb", "cc", "dd", "ee"],
                  "D": ["aaa", "bbb", "ccc", "aaa", "bbb"]})
# print(a)
a_cross = pd.crosstab(index=[a["A"], a["C"]], columns=[a["B"], a["D"]], margins=False)
print(a_cross)
print(a_cross.index)
print(a_cross.columns)
