# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: feature_histogram.py
@time: 2017/7/2 21:49
@change_time:
1.2017/7/2 21:49 构造featureHistogram结构
"""
from py_lightgbm.tree.split_info import SplitInfo
from py_lightgbm.utils import const


class FeatureEntryMeta(object):
    def __init__(self):
        self.sum_hessians = 0
        self.sum_gradients = 0
        self.cnt = 0
        return

    def __str__(self):
        repr_str = "cnt:{0}, h:{1}, g:{2}".format(
            self.cnt,
            self.sum_hessians,
            self.sum_gradients,
        )
        return repr_str

    def __repr__(self):
        return self.__str__()


class FeatureHistogram(object):
    """
        used to construct and store a histogram for a feature.
    """

    def __init__(self, feature_index, bin_mapper):
        self._meta = None
        self._is_splittable = True
        self.find_best_threshold_fun_ = self.find_best_threshold_sequence

        self._feature_index = feature_index
        self._bin_mapper = bin_mapper

        self._bin_entry = [FeatureEntryMeta() for x in xrange(len(self._bin_mapper))]

        self._min_gain_split = 0.01     # TODO: set min_gain_split
        return

    def __str__(self):
        repr_str = "index:{0}, bin_mapper:{1}".format(self._feature_index, self._bin_mapper)
        return repr_str

    def __repr__(self):
        return self.__str__()

    def init(self, train_X, data_indices, ordered_gradients, ordered_hessians):
        # build feature histogram
        for data_index in data_indices:
            value = train_X[data_index, self._feature_index]
            bin = self._bin_mapper.find_bin_idx(value)
            if bin < 0:
                continue

            self._bin_entry[bin].sum_gradients += ordered_gradients[data_index]
            self._bin_entry[bin].sum_hessians += ordered_hessians[data_index]
            self._bin_entry[bin].cnt += 1
        return

    def __sub__(self, other):
        return

    def find_best_threshold(self, sum_gradient, sum_hessian, num_data):
        # 根据best_threshold找到最佳值
        split_info = SplitInfo()

        best_sum_left_gradients = 0
        best_sum_left_hessians = const.Epsion
        best_gain = 0
        best_left_count = 0
        best_threshold_bin = 0

        sum_left_gradients = 0
        sum_left_hessians = const.Epsion
        left_count = 0

        for bin in xrange(len(self._bin_entry)):
            sum_left_gradients += self._bin_entry[bin].sum_gradients
            sum_left_hessians += self._bin_entry[bin].sum_hessians
            left_count += self._bin_entry[bin].cnt

            sum_right_gradients = sum_gradient - sum_left_gradients
            sum_right_hessians = sum_hessian - sum_left_hessians

            current_gain = self.get_leaf_split_gain(sum_left_gradients, sum_left_hessians, 1, 1) +\
                self.get_leaf_split_gain(sum_right_gradients, sum_right_hessians, 1, 1)

            if current_gain > best_gain:
                best_sum_left_gradients = sum_left_gradients
                best_sum_left_hessians = sum_left_hessians
                best_gain = current_gain
                best_left_count = left_count
                best_threshold_bin = bin

        split_info.threshold_bin = best_threshold_bin
        split_info.feature_index = self._feature_index
        split_info.gain = best_gain

        split_info.left_count = best_left_count
        split_info.right_count = num_data - best_left_count

        split_info.left_sum_gradients = best_sum_left_gradients
        split_info.left_sum_hessians = best_sum_left_hessians
        split_info.right_sum_gradients = sum_gradient - best_sum_left_gradients
        split_info.right_sum_hessians = sum_hessian - best_sum_left_hessians

        split_info.left_output = self.get_splitted_leaf_output(
            best_sum_left_gradients,
            best_sum_left_hessians,
            1,
            1)
        split_info.right_output = self.get_splitted_leaf_output(
            sum_gradient - best_sum_left_gradients,
            sum_hessian - best_sum_left_hessians,
            1,
            1,
        )

        return split_info

    def find_best_threshold_numerical(self, sum_gradient, sum_hessian, num_data):
        return

    def find_best_threshold_categorical(self, sum_gradient, sum_hessian, num_data):
        return

    def get_leaf_split_gain(self, sum_gradient, sum_hessian, l1, l2):
        abs_sum_gradients = abs(sum_gradient)
        reg_abs_sum_gradients = max(0.0, abs_sum_gradients - l1)
        return (reg_abs_sum_gradients * reg_abs_sum_gradients) / (sum_hessian + l2)

    def get_splitted_leaf_output(self, sum_gradient, sum_hessian, l1, l2):
        abs_sum_gradients = abs(sum_gradient)
        reg_abs_sum_gradients = max(0.0, abs_sum_gradients - l1)
        if sum_hessian > 0:
            return reg_abs_sum_gradients / (sum_hessian + l2)
        else:
            return -reg_abs_sum_gradients / (sum_hessian + l2)

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
