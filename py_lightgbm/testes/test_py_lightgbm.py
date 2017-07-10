# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: test_py_lightgbm.py
@time: 2017/7/9 10:34
@change_time:
1.2017/7/9 10:34
"""

import pandas as pd
import numpy as np
import py_lightgbm as lgb
from sklearn import model_selection


DATA_PATH = "../../data/mnist_train.csv"


def main():
    mnist_train = pd.read_csv(DATA_PATH, nrows=1000)
    X = mnist_train[(mnist_train.label == 6) | (mnist_train.label == 8)]
    y = X['label'].values
    X = X.drop('label', axis=1).values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 50,
        'max_depth': 10,
        'learning_rate': 0.1,
        # 'reg_lambda': 0.7,
        'n_estimators': 100,
        # 'silent': True
    }

    print('light GBM train :-)')
    clf = lgb.LGBMClassifier(**params)
    result_base = clf.fit(X_train, y_train)
    # score = result_base.score(X_test, y_test)
    # print "score:", score


if __name__ == '__main__':
    main()
