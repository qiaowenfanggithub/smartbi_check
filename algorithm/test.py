# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/9 10:33 下午

--------------------------------------------------------

"""
import pandas as pd

a = pd.DataFrame({"level": [1, 2, 2, 1, 2], "value": [6, 7, 8, 9, 10]})
print(a.groupby(["level"]).mean())

