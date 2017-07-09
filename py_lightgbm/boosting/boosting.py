# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: boosting.py
@time: 2017/7/1 11:29
@change_time:
1.2017/7/1 11:29
"""
from py_lightgbm.boosting.gbdt import Gbdt


class Booster(object):
    """
    各种实现方式的封装类
    """
    def __init__(self):
        self._booster = Gbdt()
        return

    def init(self):
        self._booster.init()
        return

    def train_one_iter(self, train_data, gradients, hessians):
        self._booster.train_one_iter(train_data)
        return


if __name__ == '__main__':
    pass
