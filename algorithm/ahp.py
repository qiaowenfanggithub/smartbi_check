# -*- coding = 'utf-8' -*-
"""
File Name : AHP.py

Description : 层次分析法样例, 比如选干部, 第一层5个因素（方面）考察，第二层3个因素（候选人），输出最后一层因素对总目标的综合得分，供决策！

Author : wangliang

Date : 2020/8/24 15:33
"""

import csv
import numpy as np


def getMat_from_file(filepath):
    l0 = []
    l1 = []
    li = []
    with open(filepath) as f:
        reader = csv.reader(f)
        for line in reader:
            line = [x for x in line if x]
            if not line:
                l1.append(l0)
                l0 = []
                continue
            for i in line:
                if "/" in i:
                    b = i.split("/")
                    li.append(int(b[0]) / float(b[1]))
                else:
                    li.append(float(i))
            l0.append(li)
            li = []
        l1.append(l0)
    return l1


def outputScore(filepath, score):
    file = open(filepath, 'w')
    for i in score:
        file.write(str(i))
        file.write("\n")
    print(score)



class AHP:
    def __init__(self, array):
        self.Mat = array
        self.row = len(array)
        self.col = len(array[0])

    def get_tezheng(self):  # 获取特征值和特征向量
        te_val, te_vector = np.linalg.eig(self.Mat)
        list1 = list(te_val)
        # print("特征值为：", te_val)
        # print("特征向量为：", te_vector)
        # 得到最大特征值对应的特征向量
        max_val = np.max(list1)
        index = list1.index(max_val)
        max_vector = te_vector[:, index]
        # print("最大的特征值:" + str(max_val) + "   对应的特征向量为：" + str(max_vector))
        return max_val, max_vector

    def get_lambdaMax(self):
        lambdaMax = self.get_tezheng()[0]
        return lambdaMax.real

    def get_Weight(self):
        max_vec = self.get_tezheng()[1]
        Weight = [x.real for x in max_vec]
        sum0 = np.sum(Weight)
        Weight = [y / sum0 for y in Weight]

        # return np.array(Weight)
        return Weight

    def RImatrix(n):  # 建立RI矩阵
        # print(n)
        n1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        n2 = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45]
        d = dict(zip(n1, n2))
        # print("该矩阵在一致性检测时采用的RI值为：", d[n])
        return d[n]

    def test_consitstence(self, max_val, RI):  # 测试一致性
        CI = (max_val - self.row) / (self.row - 1)
        if RI == 0:
            # print("判断矩阵的RI值为  " + str(0) + "  通过一致性检验")
            return True
        else:
            CR = CI / RI
            if CR < 0.10:
                # print("判断矩阵的CR值为  " + str(CR) + "  通过一致性检验")
                return True

            else:
                # print("判断矩阵的CR值为  " + str(CR) + "  判断矩阵未通过一致性检验，请重新输入判断矩阵")
                return False


def main():
    l1 = getMat_from_file(r"./data/data2.csv")
    l2 = getMat_from_file(r"./data/data3.csv")

    l1_factors, l2_factors = len(l1[0]), len(l2[0])
    if not l1:
        print("read the first layer Matrix fail, please check")
        return
    if l1_factors != len(l2):
        print("the second layer's Matrix Number not equal the first layer Factors")
        return

    l1_Weight = []
    for mat in l1:
        a = AHP(mat)
        w = a.get_Weight()
        l1_Weight.append(w)

    l2_Weight = []
    for mat in l2:
        a = AHP(mat)
        w = a.get_Weight()
        l2_Weight.append(w)
    # output score and write into file
    score = np.dot(np.array(l2_Weight).T, np.array(l1_Weight).T)
    outputScore("./sc.txt", score)
    print("ok")


if __name__ == "__main__":
    main()
