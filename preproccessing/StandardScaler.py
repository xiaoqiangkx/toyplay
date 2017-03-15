# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: StandardScaler.py
@time: 2017/3/15 22:38
@change_time:
1.2017/3/15 22:38
"""
import numpy as np
import math


class StandardScaler(object):
    def __init__(self):
       return

    def fit_transform(self, X):
        """
        按照行进行数据的转换
        """
        rows, cols = X.shape
        result = np.zeros(X.shape)
        for row in xrange(rows):
            x_mean = np.mean(X[row, :])
            x_var = np.var(X[row, :])
            result[row, :] = (X[row, :] - x_mean) / math.sqrt(x_var)
        return result


if __name__ == '__main__':
    data = np.array([[1, 2, 3], [2, 2, 3]])
    result = StandardScaler().fit_transform(data)
    print result