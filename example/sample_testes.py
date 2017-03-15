# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: sample_testes.py
@time: 2017/3/15 20:59
@change_time:
1.2017/3/15 20:59
"""
from logistic.logistic import LogisticRegression
from utils import file_utils as FU
from utils import formula as FORMULA
from utils import sample as SAMPLE
from matplotlib import pyplot as plt
from preproccessing.StandardScaler import StandardScaler
from utils import logger as LOGGER


def model_test(X, Y, method):
    amount = 0
    times = 0
    for cv_data, cv_target, test_data, test_target in SAMPLE.iter_sample_data(X, Y, method):
        times += 1
        model = LogisticRegression(delta=0.01, alpha=0.01)
        model.fit(cv_data, cv_target)
        predict_y = model.predict(test_data)
        amount += FORMULA.cal(predict_y, test_target)

    return float(amount) / times

if __name__ == '__main__':
    filename = "../data/iris.csv"
    X, Y = FU.load_iris_data(filename)
    X = StandardScaler().fit_transform(X)
    X = FORMULA.plus_one(X)

    LOGGER.setLevel(LOGGER.LEVEL_NORMAL)
    print u"10折交叉法:", model_test(X, Y, 10)
    print u"留一法:", model_test(X, Y, 1)

    # model.draw_data(X, Y)
    # model.draw_line(X)
    # model.draw_loss()
    # plt.show()
