# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: bin.py
@time: 2017/7/8 11:58
@change_time:
1.2017/7/8 11:58
"""

TYPE_CATEGORY = 1
TYPE_NUMERICAL = 2


class HistogramBinEntry(object):

    """
    store data for one histogram bin
    """
    def __init__(self):
        self._sum_gradients = 0
        self._sum_hessians = 0
        self._cnt = 0
        return


class BinMapper(object):
    """
    Convert feature values into bin and store some meta information for bin
    Every feature will get a BinMapper
    """

    def __init__(self):
        self._num_bins = 0              # number of bins
        self._bin_upper_bound = []
        self._bin_type = TYPE_NUMERICAL
        self._category2bin = {}         # int to unsigned int
        self._bin2category = {}

        self._is_trival = False

        self._min_value = 0
        self._max_value = 0
        self._default_bin = 0
        return

    def find_bin(self, values, num_values, total_sample_cnt, max_bin, min_data_in_bin, min_split_data, bin_type):
        """
        Construct feature values to binMapper
        :param values:
        :param num_values:
        :param total_sample_cnt:
        :param max_bin:
        :param min_data_in_bin:
        :param min_split_data:
        :param bin_type:
        :return:
        """

        # 1. find distinct values first

        # 2. push zero into distinct values

        # 3. deal with numerical and categorical data

        return

    def greedy_find_bin(self, distinct_values, counts, num_distinct_values, max_bin, total_cnt, min_data_in_bin):

        return


class OrderedBin(object):
    pass




if __name__ == '__main__':
    pass
