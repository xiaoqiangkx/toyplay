# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: tree.py
@time: 2017/7/2 15:56
@change_time:
1.2017/7/2 15:56
"""


class Tree(object):

    def __init__(self, max_num_leaves):
        self.max_leaves = max_num_leaves

        self.num_leaves = 0

        # used fo non-leaf node
        self.left_child = [-1] * (self.max_leaves - 1)
        self.right_child = [-1] * (self.max_leaves - 1)
        self.split_feature_index = [-1] * (self.max_leaves - 1)
        self.threshold_in_bin = [-1] * (self.max_leaves - 1)      # threshold in bin
        self.split_gain = [0] * (self.max_leaves - 1)

        self.internal_values = [0] * (self.max_leaves - 1)
        self.internal_counts = [0] * (self.max_leaves - 1)

        # used for leaf node
        self.leaf_parent = [-1] * self.max_leaves
        self.leaf_values = [0] * self.max_leaves
        self.leaf_counts = [0] * self.max_leaves

        self.leaf_depth = [0] * self.max_leaves            # depth for leaves

        # root is in the depth 0
        self.leaf_depth[0] = 0
        self.num_leaves = 1
        self.leaf_parent[0] = -1
        return

    def split(self, leaf, split_info):
        """
        :param best_leaf: the index of best leaf
        :param best_split_info: split info
        :return:
        """
        new_node_idx = self.num_leaves - 1

        self.split_feature_index[new_node_idx] = split_info.feature_index
        self.threshold_in_bin[new_node_idx] = split_info.threshold
        self.split_gain[new_node_idx] = split_info.gain
        self.left_child[new_node_idx] = 
        return


    def predict(self, feature_values):

        return


if __name__ == '__main__':
    pass
