# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: test_dataset.py
@time: 2017/7/9 16:14
@change_time:
1.2017/7/9 16:14
"""
import py_lightgbm as lgb
import numpy as np
import random


def main():
    params = {
        "max_bin": 10,
    }
    clf = lgb.LGBMClassifier(**params)
    X_train = np.zeros((100, 2))
    y_train = np.zeros((100, 1))
    for i in xrange(100):
        X_train[i, 0] = random.randint(0, 10)
        X_train[i, 1] = random.randint(0, 20)

    clf.fit(X_train, y_train)
    clf.print_bin_mappers()
    return


if __name__ == "__main__":
    main()
