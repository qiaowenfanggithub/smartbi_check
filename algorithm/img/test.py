# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : test

Description : 

Author : leiliang

Date : 2020/7/7 9:47 上午

--------------------------------------------------------

"""
import os
from PIL import Image
import numpy as np


def init(fileList, n):
    for i in range(2):
        img = Image.open(str(fileList[i]))
        new_img = img.resize((32, 32))
        w, h = new_img.size
        with open(str(i) + "_9.txt", "w") as f:
            for c in range(h):
                for j in range(w):
                    f.write(str(int((255 - np.array((new_img.getpixel((j, c))))) / 255)))
                    if j == w - 1:
                        f.write("\n")
        f.close()


if __name__ == "__main__":
    # fileList = os.listdir("9.png")

    fileList = ['9.png']
    n = len(fileList)
    init(fileList, n)
