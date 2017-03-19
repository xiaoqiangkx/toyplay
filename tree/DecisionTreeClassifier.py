# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: DecisionTreeClassifier.py
@time: 2017/3/18 10:51
@change_time:
1.2017/3/18 10:51
"""
from algorithm import DecisionTree as DT
import numpy as np


class DecisionTreeClassifier(object):

    def __init__(self, depth=5):
        self.tree = DT.DecisionTree(depth=depth)
        return

    def fit(self, data, target, choose_func=DT.CHOOSE_INFO_ENTROPY):
        self.tree.make_tree(data, target, choose_func)
        return

    def show(self):
        self.tree.show()
        return

    def save(self, filename):
        self.tree.save(filename)
        return

    def predict(self, data):
        m, n = data.shape
        target = np.zeros(m)
        for idx, item in enumerate(data):
            result = self.tree.decide(item)
            target[idx] = result

        return target
