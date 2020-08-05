# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/9 10:33 下午

--------------------------------------------------------

"""
table = "table"
key_list = [["a", "b", "c"]]
value_list = [["1", "2", "3"]]
for key in key_list:
    for value in value_list:
        sql = "INSERT INTO {}({}) VALUES ('{}')".format(table, ",".join(key), "','".join(value))
        print(sql)
