# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: formula.py
@time: 2017/3/13 21:55
@change_time:
1.2017/3/13 21:55
"""
import numpy as np
from utils import logger as LOGGER
import operator


def sigmoid(x, w):
    return 1.0 / (1.0 + np.exp(np.dot(-w.T, x).astype(float)))


def cal(result, test_target_data):
    data = (result == test_target_data)
    count = 0
    for elem in data[0]:
        if elem:
            count += 1
    amount = len(test_target_data[0])
    precision = float(count) / amount
    LOGGER.info("count({0})/num({1}), precision:{2}".format(count, amount, precision))
    return precision


def plus_one(X):
    m, n = X.shape
    X = np.row_stack((X, np.ones(n)))
    return X


# 仿照STL封装一些常用的operator方法,方便以后扩展 #
class MyOerator(object):
    def __init__(self):
        pass

class lte(MyOerator):
    def __init__(self, a):
        super(lte, self).__init__()
        self.a = a
        return

    def __call__(self, *args, **kwargs):
        arg_list = list(args)
        arg_list.append(self.a)
        return operator.le(*args, **kwargs)

    def __repr__(self):
        return "lte({0})".format(self.a)


class equal(MyOerator):
    def __init__(self, a):
        super(equal, self).__init__()
        self.a = a
        return

    def __call__(self, *args, **kwargs):
        arg_list = list(args)
        arg_list.append(self.a)
        return operator.eq(*args, **kwargs)

    def __repr__(self):
        return "equal({0})".format(self.a)
