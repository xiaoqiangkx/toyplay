# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: decision_tree_test.py
@time: 2017/3/18 16:22
@change_time:
1.2017/3/18 16:22
"""
from algorithm import DecisionTree as DT
import random
import numpy as np


def make_dataset(num, num_feature, category=3):
    """
    1. make a dataset with the amount of num
    and the amount of num_feature features
    2. every sample is a different category
    """
    data = np.random.randint(1, 5, size=(num, num_feature))
    target = np.random.randint(1, category, size=(1, num))[0]
    return data, target


if __name__ == '__main__':
    # Make a simple Decision Tree and plot it
    num = 100
    num_feature = 10
    data, target = make_dataset(num, num_feature, category=10)
    # print target
    decision_tree = DT.DecisionTree()
    decision_tree.make_tree(data, target, choose_func=DT.CHOOSE_INFO_ENTROPY)
    decision_tree.show()
    dot_tree = decision_tree.save("test.dot")
    # print dot_tree.source
