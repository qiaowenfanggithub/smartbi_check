# -*- coding = 'utf-8' -*-
"""

--------------------------------------------------------

File Name : base64_to_png

Description : 

Author : leiliang

Date : 2020/7/27 2:38 下午

--------------------------------------------------------

"""
import base64
import os
import time


def base64_to_img(base64_str):
    if not os.path.exists("./img"):
        os.makedirs("./img")
    with open('./img/{}.jpg'.format(time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())), 'wb') as file:
        jiema = base64.b64decode(base64_str)  # 解码
        file.write(jiema)  # 将解码得到的数据写入到图片中


if __name__ == '__main__':
    data = "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MZubm82SkhLzt7/9bSG6kLFoNGqOGzfOvPfeewe959TveJ90+u6U7/mtt95qjh071nzuuefM/fv3m62trWZVVZV5zz339Mc49XueLxSALjKwAJw5c6Z566239v+8ZMkSc9y4ceaIESPM0aNHm3PmzDHb29sLkGl2brzxRrOmpsYcMWKEOXbsWPPGG280Ozs7+98f2G/TNM1f/vKXZkNDgzlixAjzkksuMX/zm99YnHX20u23Uz5v0zTN//t//685efJk0+v1mpMmTTJ//OMfx71/6623mjNnzozbtmXLFnPKlCnmiBEjzE984hPmz372M+sSzqF0+/7DH/7QnDBhglleXm6ef/755qxZs8zf/e53FmedvRdffNGUZO7bt2/Qe079jvdJp+9O+Z6Hw2GzubnZHDdunFleXm5+4hOfML/1rW+ZkUikP8bJ3/N88JjmENNoAwAAwFG4BxAAAMBlKAABAABchgIQAADAZSgAAQAAXIYCEAAAwGUoAAEAAFyGAhAAAMBlKAABAABchgIQAADAZSgAAQAAXIYCEAAAwGUoAAEAAFyGAhAAAMBlKAABAABchgIQAADAZSgAAQAAXIYCEAAAwGUoAAEAAFyGAhAAAMBlKAABAABchgIQAADAZSgAAQAAXIYCEAAAwGUoAAEAAFyGAhAAAMBlKAABAABchgIQAADAZSgAAQAAXIYCEACA/99uHQgAAAAACPK3HuSiCGYEEABgRgABAGYCWE3npb/L4joAAAAASUVORK5CYII="
    base64_to_img(data)
