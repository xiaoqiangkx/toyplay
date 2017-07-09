# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: dataset.py
@time: 2017/7/8 10:59
@change_time:
1.2017/7/8 10:59
"""


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
    def __init__(self):
        self._feature_group = []        # ?? Store feature group information
        self._used_feature_map = {}     # ?? real feature index to used index

        self._num_features = 0          # number of used feature
        self._num_total_features = 0    # number of total feature
        self._num_data = 0              # number of data
        self._metadata = None           # store brief label of MetaData
        self._label_idx = 0             # index of label

        self._feature_names = []        # feature names
        self._num_groups = 0

        self._real_feature_idx = {}
        self._feature2group = {}
        self._group_bin_boundary = {}
        self._group_feature_start = {}
        self._group_feature_cnt = {}
        return

    def create_ordered_bins(self, order_bins):
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


if __name__ == '__main__':
    pass
