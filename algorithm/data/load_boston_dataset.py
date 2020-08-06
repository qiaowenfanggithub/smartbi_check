# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : load_boston_dataset

Description : 

Author : leiliang

Date : 2020/8/6 3:00 下午

--------------------------------------------------------

"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
feature = boston["data"]
target = boston["target"].reshape(-1, 1)
data = pd.DataFrame(np.hstack((feature, target)), columns=boston["feature_names"].tolist() + ["house_prices"])
print(data)
data.to_csv("boston.csv")