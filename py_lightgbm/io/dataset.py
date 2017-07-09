# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: dataset.py
@time: 2017/7/8 10:59
@change_time:
1.2017/7/8 10:59 init dataset, add BinMapper
"""
from py_lightgbm.io.bin import BinMapper


MIN_DATA_IN_BIN = 10


class MetaData(object):
    """
    store some meta data used for trainning data
    """
    def __init__(self):
        self._labels = []
        self._init_scores = []
        return


class Dataset(object):
    """
    The main class of dataset
    """
    def __init__(self, X, y, feature_name):
        self._train_data = X
        self._feature_names = feature_name
        self._labels = y
        print self._train_data.shape
        self._num_data = self._train_data.shape[0]
        self._num_feature = self._train_data.shape[1]
        return

    def construct_histograms(self, is_feature_used, data_indices, num_data, leaf_idx, ordered_bins,
                             gradients, hessians, ordered_gradients, ordered_hessians, is_const_hessian,
                             histogram_data):
        return

    def fix_histogram(self, feature_idx, sum_gradients, sum_hessian, num_data, histogram_data):

        return

    def split(self, feature, threshold, default_bin_for_zero, data_indices, num_data, lte_indices, gt_indices):

        return

    def construct(self, bin_mappers, sample_non_zero_indices, num_per_col, total_sample_cnt, io_config):

        return

    def create_bin_mapper(self, max_bin):
        bin_mappers = []
        for i in xrange(self._num_feature):
            bin_mapper = BinMapper()
            values = self._train_data[:, i]
            bin_mapper.find_bin(values, max_bin, min_data_in_bin=MIN_DATA_IN_BIN)
            bin_mappers.append(bin_mapper)
        return bin_mappers


if __name__ == '__main__':
    pass
