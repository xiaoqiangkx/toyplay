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
import py_lightgbm as lgb
from sklearn import model_selection
from collections import Counter
from sklearn import metrics
import profile
import time
from py_lightgbm.logmanager import logger

DATA_PATH = "../../data/mnist_train.csv"

_LOGGER = logger.get_logger("Test")


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
        'n_estimators': 1,
        # 'silent': True
    }

    _LOGGER.critical('light GBM train :-)')
    clf = lgb.LGBMClassifier(**params)
    _LOGGER.info("data_shape: {0}, {1}".format(X_train.shape, Counter(y_train)))
    _LOGGER.info("y_test:{0}".format(y_test))

    profile.enable_profile()
    timestamp_start = time.time()
    _LOGGER.critical("starting profile")
    clf.fit(X_train, y_train)
    # clf.show()
    y_predict = clf.predict_proba(X_test)
    _LOGGER.info(y_predict)
    score = metrics.accuracy_score(y_test, y_predict)
    _LOGGER.critical("score:{0}".format(score))
    profile.close_profile()
    timestamp_end = time.time()
    _LOGGER.critical("finish profile:{0}".format(timestamp_end - timestamp_start))


if __name__ == '__main__':
    main()
