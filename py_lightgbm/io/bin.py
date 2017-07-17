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
from py_lightgbm.utils import const

import math


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

        self._bin_type = const.TYPE_NUMERICAL

        self._category2bin = {}         # int to unsigned int
        self._bin2category = {}

        self._is_trivial = False

        self._min_value = 0
        self._max_value = 0
        self._default_bin = 0
        return

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        repr_str = "nb:{0}, bound:{1}".format(self._num_bins, self._bin_upper_bound)
        return repr_str

    def __len__(self):
        return len(self._bin_upper_bound)

    def upper_at(self, bin_index):
        return self._bin_upper_bound[bin_index]

    def find_lower_bound(self, threshold):
        print self._bin_upper_bound
        index = self._bin_upper_bound.index(threshold)

        if index == 0:
            return -float("inf")

        return self._bin_upper_bound[index - 1]

    def find_bin(self, values, max_bin, bin_type=const.TYPE_NUMERICAL, min_data_in_bin=0, min_split_data=0):
        """
        Construct feature values to binMapper
        """
        self._bin_type = bin_type

        if bin_type is const.TYPE_NUMERICAL:
            distinct_values = sorted(list(set(values)))     # set操作会默认进行排序操作
            self._min_value = distinct_values[0]
            self._max_value = distinct_values[-1]
            counts = Counter(values)
        else:    # TYPE_NUMERICAL, fill category2bin and bin2category
            category_counts = Counter(values)
            bin_num = 1
            counts = {}
            max_bin = min(max_bin, int(math.ceil(len(category_counts) * 0.98)))

            for key, cnt in category_counts.most_common(max_bin):
                self._category2bin[key] = bin_num
                self._bin2category[bin_num] = key
                counts[bin_num] = cnt
                bin_num += 1

            distinct_values = self._category2bin.values()

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

    def find_bin_idx(self, value):
        # find the bin for value, use bi_search
        if self._bin_type is const.TYPE_CATEGORY:
            value = self._category2bin.get(value, float("inf"))

        st = 0
        data = self._bin_upper_bound
        ed = len(data) - 1

        mid = -1
        while st <= ed:
            mid = (st + ed) / 2
            if data[mid] == value:
                break
            elif data[mid] < value:
                st = mid + 1
            else:
                ed = mid - 1

        return mid


class OrderedBin(object):
    pass


if __name__ == '__main__':
    bin_mapper = BinMapper()
    bin_mapper._bin_upper_bound = [0, 1, 3, 4, 5, float("inf")]
    result = bin_mapper.find_bin_idx(3.5)
    print result
