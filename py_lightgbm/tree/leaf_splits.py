# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: leaf_splits.py
@time: 2017/7/10 21:30
@change_time:
1.2017/7/10 21:30
"""


class LeafSplits(object):
    def __init__(self, num_data):
        self.leaf_index = -1
        self.num_data_in_leaf = num_data
        self.num_data = num_data
        self.sum_gradients = 0
        self.sum_hessians = 0
        self.data_indices = range(num_data)
        return

    def init(self, gradients, hessians):
        self.leaf_index = 0
        self.sum_gradients = sum(gradients)
        self.sum_hessians = sum(hessians)
        return

    def init_with_data_partition(self, leaf, data_partition, gradients, hessians):
        self.leaf_index = leaf

        self.data_indices = data_partition.get_indices_of_leaf(leaf)
        self.num_data_in_leaf = len(self.data_indices)

        self.sum_gradients = 0
        self.sum_hessians = 0

        for index in self.data_indices:
            self.sum_gradients += gradients[index]
            self.sum_hessians += hessians[index]
        return

    def reset(self):
        self.leaf_index = -1
        self.sum_gradients = 0
        self.sum_hessians = 0
        self.data_indices = []
        return
