# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: data_partition.py
@time: 2017/7/10 21:08
@change_time:
1.2017/7/10 21:08
"""


class DataPartition(object):

    def __init__(self, num_data, num_leaves):
        self.num_data = num_data
        self.num_leaves = num_leaves

        # the three list below used for data_partition
        self._leaf_begin = None
        self._leaf_count = None
        self._indices = None

        self._temp_left_indices = None
        self._temp_right_indices = None
        return

    def init(self):
        self._leaf_begin = [0] * self.num_leaves
        self._leaf_count = [0] * self.num_leaves
        self._indices = range(self.num_data)

        self._temp_left_indices = [0] * self.num_data
        self._temp_right_indices = [0] * self.num_data
        self._leaf_count[0] = self.num_data
        return

    def get_indices_of_leaf(self, leaf):
        begin = self._leaf_begin[leaf]
        cnt = self._leaf_count[leaf]

        return self._indices[begin:begin+cnt]

    def split(self, left_leaf, train_data, feature_index, threshold, right_leaf):

        begin = self._leaf_begin[left_leaf]
        cnt = self._leaf_count[left_leaf]

        # 将这个leaf划分为两份
        left_indices, right_indices = train_data.split(feature_index, threshold, self._indices, begin, begin + cnt)
        left_cnt = len(left_indices)

        self._indices[begin:begin+left_cnt] = left_indices
        self._indices[begin+left_cnt:begin+cnt] = right_indices
        # 增加新的leaf的split信息

        self._leaf_count[left_leaf] = left_cnt

        self._leaf_begin[right_leaf] = begin + left_cnt
        self._leaf_count[right_leaf] = cnt - left_cnt

        print self._leaf_begin[left_leaf],
        return


