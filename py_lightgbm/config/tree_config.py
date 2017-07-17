# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: tree_config.py
@time: 2017/7/2 22:38
@change_time:
1.2017/7/2 22:38
"""
from py_lightgbm.utils import const


class TreeConfig(object):
    def __init__(self):
        # 性能参数
        self.num_leaves = const.DEFAULT_NUM_LEAVES
        self.max_depth = const.DEFAULT_MAX_DEPTH
        self.learning_rate = const.DEFAULT_LEARNING_RATE
        self.n_estimators = const.DEFAULT_NUM_ESTIMATORS
        self.max_bin = const.DEFAULT_MAX_BIN
        self.min_split_gain = const.DEFAULT_MIN_SPLIT_GAIN
        self.reg_alpha = const.DEFAULT_REG_ALPHA
        self.reg_lambda = const.DEFAULT_REG_LAMBDA
        self.min_child_samples = const.DEFAULT_MIN_CHILD_SAMPLES
        return


if __name__ == '__main__':
    pass
