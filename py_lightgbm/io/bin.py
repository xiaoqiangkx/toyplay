# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: bin.py
@time: 2017/7/8 11:58
@change_time:
1.2017/7/8 11:58
"""
from collections import Counter


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

        self._is_trivial = False

        self._min_value = 0
        self._max_value = 0
        self._default_bin = 0
        return

    def find_bin(self, values, max_bin, bin_type=TYPE_NUMERICAL, min_data_in_bin=0, min_split_data=0):
        """
        Construct feature values to binMapper
        """
        distinct_values = list(set(values))     # set操作会默认进行排序操作
        self._min_value = distinct_values[0]
        self._max_value = distinct_values[-1]
        counts = Counter(values)

        self._bin_upper_bound = self.greedy_find_bin(
            distinct_values, counts, max_bin,
            min_data_in_bin=min_data_in_bin,
        )

        self._num_bins = len(self._bin_upper_bound)
        return

    def greedy_find_bin(self, distinct_values, counts, max_bin, min_data_in_bin=0):
        # update upper_bound
        num_total_values = len(distinct_values)
        bin_upper_bound = []

        mean_data_in_bin = num_total_values / max_bin
        mean_data_in_bin = max(min_data_in_bin, mean_data_in_bin)
        mean_data_in_bin = max(1, mean_data_in_bin)

        cnt = 0
        for value in distinct_values:
            num_value = counts[value]
            cnt += num_value

            if cnt >= mean_data_in_bin:
                bin_upper_bound.append(value)
                cnt = 0

        bin_upper_bound.append(float("inf"))
        return bin_upper_bound


class OrderedBin(object):
    pass


if __name__ == '__main__':
    pass
