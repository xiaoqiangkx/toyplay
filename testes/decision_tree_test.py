# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: decision_tree_test.py
@time: 2017/3/18 16:22
@change_time:
1.2017/3/18 16:22
"""
from algorithm.DecisionTree import DecisionTree
import random


def make_dataset(num, num_feature):
    """
    1. make a dataset with the amount of num
    and the amount of num_feature features
    2. every sample is a different category
    """
    data = []
    for i in xrange(num):
        temp = [random.randint(1, 5) for j in xrange(num_feature)]
        data.append(temp)

    return data


def make_tree(data):
    """
    rule: choose the first remaining features left
    """
    tree = DecisionTree()
    tree.make_tree(data)
    return tree


if __name__ == '__main__':
    # Make a simple Decision Tree and plot it
    num = 10
    num_feature = 5
    data = make_dataset(num, num_feature)
    decision_tree = make_tree(data)
    decision_tree.show()
    dot_tree = decision_tree.save("test.dot")
    # print dot_tree.source
