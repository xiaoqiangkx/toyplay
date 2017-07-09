# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: feature_histogram.py
@time: 2017/7/2 21:49
@change_time:
1.2017/7/2 21:49 构造featureHistogram结构
"""


class FeatureHistogram(object):
    """
        used to construct and store a histogram for a feature.
    """

    def __init__(self):
        self._data = None
        self._meta = None
        self._is_splittable = True
        self.find_best_threshold_fun_ = None
        return

    def __sub__(self, other):
        return

    def find_best_threshold(self, sum_gradient, sum_hessian, num_data):
        return

    def find_best_threshold_numerical(self, sum_gradient, sum_hessian, num_data):
        return

    def find_best_threshold_categorical(self, sum_gradient, sum_hessian, num_data):
        return

    def get_leaf_split_gain(self, sum_gradient, sum_hessian, l1, l2):
        return

    def get_splitted_leaf_output(self, sum_gradient, sum_hessian, l1, l2):
        return

    def find_best_threshold_sequence(self, sum_gradient, sum_hessian, num_data, min_gain_shift):
        pass


class HistogramPool(object):
    def __init__(self):
        self._pool = []
        self._data = []
        self._feature_metas = []

        self._cache_size = 0
        self._total_size = 0
        self._is_enough = False

        self._mapper = []
        self._inverse_mapper = []
        self._last_used_time = []
        self._cur_time = 0
        return

    def move(self):
        return

    def get(self):
        return

    def reset_config(self):
        return

    def dynamic_change_size(self, train_data, tree_config, cache_size, total_size):
        return

    def reset_map(self):
        return

    def reset(self):
        return


if __name__ == '__main__':
    pass
