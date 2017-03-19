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
from utils import logger
import operator
from collections import Counter


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
    logger.info("count({0})/num({1}), precision:{2}".format(count, amount, precision))
    return precision


def cal_new(result, test_target_data):
    data = (result == test_target_data)
    count = 0
    for elem in data:
        if elem:
            count += 1
    amount = len(test_target_data)
    precision = float(count) / amount
    logger.info("count({0})/num({1}), precision:{2}".format(count, amount, precision))
    return precision


def plus_one(X):
    m, n = X.shape
    X = np.row_stack((X, np.ones(n)))
    return X


def calculate_entropy(target, index_list):
    cnt = Counter()
    for index in index_list:
        cnt[target[index]] += 1

    result = 0
    for num in cnt.itervalues():
        prob = float(num)/len(index_list)
        result += prob * np.log(prob)
    return -result

def calculate_gini(target, index_list):
    cnt = Counter()
    for index in index_list:
        cnt[target[index]] += 1

    result = 0
    for num in cnt.itervalues():
        prob = float(num)/len(index_list)
        result += prob * prob
    return 1 - result


def calculate_iv(target_dict):
    result = 0
    total = sum([len(x) for x in target_dict.itervalues()])
    for index_list in target_dict.itervalues():
        prob = len(index_list) / float(total)
        result += prob * np.log(prob)

    return max(1, -result)


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
        return operator.le(*arg_list, **kwargs)

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
        return operator.eq(*arg_list, **kwargs)

    def __repr__(self):
        return "equal({0})".format(self.a)


if __name__ == '__main__':
    a = {1: [2, 3], 2:[3, 4]}
    result = calculate_iv(a)
    print result
