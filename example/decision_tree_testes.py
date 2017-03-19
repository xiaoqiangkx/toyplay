# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: decision_tree_testes.py
@time: 2017/3/19 15:39
@change_time:
1.2017/3/19 15:39
"""
from utils import file_utils
from utils import sample
from algorithm import DecisionTree as DT
from tree import DecisionTreeClassifier as DTC
from utils import logger
from utils import formula


if __name__ == '__main__':
    train_filename = "../data/mnist_train.csv"
    test_filename = "../data/mnist_test.csv"
    training_data, training_target, _, _ = file_utils.load_mnist_data(train_filename, test_filename)
    # training_data = training_data[0:100, :]
    # training_target = training_target[0:100]
    train_index, cv_index, test_index = sample.sample_target_data(training_target)

    train_data = training_data[train_index, :]
    train_target = training_target[train_index]
    cv_data = training_data[cv_index, :]
    cv_target = training_target[cv_index]
    test_data = training_data[test_index, :]
    test_target = training_target[test_index]

    tree = DTC.DecisionTreeClassifier(depth=3)
    logger.info("start making tree")
    tree.fit(train_data, train_target, choose_func=DT.CHOOSE_GAIN_RATIO)
    logger.info("finish making tree")
    # tree.show()

    logger.info("start predicting")
    cv_result = tree.predict(cv_data)
    logger.info("calculate precision")
    formula.cal_new(cv_result, cv_target)
