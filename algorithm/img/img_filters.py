# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : img_filters

Description :

    1: 1*2卷积核
    2: 3*3(中间8，周围-1)

Author : leiliang

Date : 2020/7/3 2:18 下午

--------------------------------------------------------

"""
import cv2
import numpy as np
import copy


if __name__ == '__main__':
    img = cv2.imread("../img/lina.png")

    # gray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_ori = copy.deepcopy(img)

    print(img)
    # 3*3
    # kernel = np.ones((3, 3), np.float32) / 9

    # 8 * others - 1 * center
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
    # -8 * center + 1 * others
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
    # 拉普拉斯
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

    # 1*2
    # kernel = np.array([1/10]*10, np.float32)

    # 2*1
    # kernel = np.array([[1/2], [1/2]], np.float32)

    # sobel-x
    # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    # sobel-y
    # kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    # sobel
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(x)  # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # gass
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # median
    # img = cv2.medianBlur(img, 15)
    #
    # canny
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.Canny(img, 100, 150)

    # 2D卷积，低通滤波
    # img = cv2.blur(img, (5, 5))

    # img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.bilateralFilter(img, 7, 155, 45)
    img_gauss = copy.deepcopy(img)
    img = cv2.filter2D(img_gauss, -1, kernel)
    # img = cv2.medianBlur(img, 3)

    # _, dst = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)


    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 7)
    # img = cv2.blur(img, (3, 3))
    # img = 255 - img

    img = img_gauss + img
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 3, 175, 75)
    cv2.imwrite('../img/lina_1.png', img)
    print("finish")
