# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: file_utils.py
@time: 2017/3/13 22:00
@change_time:
1.2017/3/13 22:00
"""

import pandas as pd
import numpy as np


def load_water_melon_data(filename, header='infer'):
    df = pd.read_csv(filename, header=header)
    m, n = df.shape
    X = df.values[:, :n-1]
    Y = df.values[:, n-1:n]
    return X.T, Y.T


def load_iris_data(filename, header='infer'):
    df = pd.read_csv(filename, header=header)
    m, n = df.shape
    X = df.values[:, 1:n-1]

    Y = df.values[:, n-1:]
    for i in xrange(m):
        if Y[i, 0] == 'Iris-versicolor':
            Y[i, 0] = 1
        elif Y[i, 0] == 'Iris-virginica':
            Y[i, 0] = 0
    return X.T, Y.T


def load_mnist_data(train_filename, test_filename, header="infer"):
    train_df = pd.read_csv(train_filename, header=header)
    test_df = pd.read_csv(test_filename, header=header)

    m, n = train_df.shape
    training_data = train_df.values[:, 1:n]
    training_target = train_df.values[:, 0].T

    m, n = test_df.shape
    test_data = test_df.values[:, 1:n]
    test_target = test_df.values[:, 0].T

    return training_data, training_target, test_data, test_target


if __name__ == '__main__':

    filename = "../data/iris.csv"
    X, Y = load_iris_data(filename)
    print X, Y
