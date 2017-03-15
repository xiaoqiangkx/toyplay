# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: sample.py
@time: 2017/3/15 21:32
@change_time:
1.2017/3/15 21:32
"""
import random
from collections import defaultdict


SAMPLE_AS_TEN = 10      # 10折交叉法
SAMPLE_LEAVE_ONE = 1    # 留一法


def iter_sample_data(X, Y, method=SAMPLE_AS_TEN):
    """
    默认10折交叉法和留一法
    """

    m, n = X.shape
    col_index = range(n)        # n列数据
    random.shuffle(col_index)

    for i in xrange(0, n, method):
        train_index = col_index[0: i] + col_index[i+method:]
        test_index = col_index[i:i + method]
        yield X[:, train_index], Y[:, train_index], X[:, test_index], Y[:, test_index]
    return


if __name__ == '__main__':
    from utils import file_utils as FU
    filename = "../data/iris.csv"
    X, Y = FU.load_iris_data(filename)
    for train_x, train_y, test_x, test_y in iter_sample_data(X, Y, 50):
        print train_x, train_y, test_x, test_y
