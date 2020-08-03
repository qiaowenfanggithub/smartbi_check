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
import numpy as np
import tushare as ts
from sklearn.linear_model import LinearRegression

# 使用tushare获取数据
ts.set_token('your token')
pro = ts.pro_api()


# 获取沪深300与中国平安股票2018年全年以及2019年前7个交易日的日线数据
index_hs300 = pro.index_daily(ts_code='399300.SZ', start_date='20180101', end_date='20190110')
stock_000001 = pro.query('daily', ts_code='000001.SZ', start_date='20180101', end_date='20190110', fields='ts_code, trade_date ,pct_chg')

# 保留用于回归的数据，数据对齐
# join的inner参数用于剔除指数日线数据中中国平安无交易日的数据，比如停牌。
# 同时保留中国平安和指数的日收益率并分别命名两组列名y_stock, x_index，确定因变量和自变量
index_pct_chg = index_hs300.set_index('trade_date')['pct_chg']
stock_pct_chg = stock_000001.set_index('trade_date')['pct_chg']
df = pd.concat([stock_pct_chg, index_pct_chg], keys=['y_stock', 'x_index'], join='inner', axis=1, sort=True)


# 选中2018年的x，y数据作为现有数据进行线性回归
df_existing_data = df[df.index < '20190101']

# 注意：自变量x为pandas.DataFrame类型，其为二维数据结构，注意与因变量的pandas.Series数据结构的区别。也可以用numpy.array的.reshape((-1, 1))将数据结构改为二维数组。
x = df_existing_data[['x_index']]
y = df_existing_data['y_stock']
