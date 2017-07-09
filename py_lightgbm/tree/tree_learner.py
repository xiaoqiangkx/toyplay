# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: tree.py
@time: 2017/7/1 11:26
@change_time:
1.2017/7/1 11:26
"""
from py_lightgbm.tree.tree import Tree
from py_lightgbm.tree.feature_histogram import FeatureHistogram
from py_lightgbm.tree.split_info import SplitInfo


class TreeLearner(object):
    def __init__(self, num_leaves, train_data):
        self._gradients = None
        self._hessians = None
        self._tree_config = None        # TODO: 用于管理tree config文件
        self._histogram_pool = None
        self._train_data = train_data
        self._max_cache_size = None
        self._num_leaves = num_leaves
        self._num_features = self._train_data.num_features
        self._num_data = self._train_data.num_data

        self._smaller_leaf_histogram_array = []
        self._larger_leaf_histogram_array_ = []
        self._best_split_per_leaf = {}

        self._smaller_leaf_split = None     # store the best splits for this leaf at smaller leaf
        self._larger_leaf_split = None

        self.init()
        return

    def init(self):
        # TODO
        self._histogram_pool.DynamicChangeSize(self._train_data, self._tree_config,
                                               self._max_cache_size, self._tree_config.num_leaves)
        return

    def train(self, gradients, hessians):
        self._gradients = gradients
        self._hessians = hessians

        self.before_train()

        new_tree = Tree(self._num_leaves)

        left_leaf = 0
        right_leaf = -1
        cur_depth = 1

        for split in xrange(self._num_leaves - 1):
            if not self.before_find_best_leave(new_tree, left_leaf, right_leaf):    # 检查数据
                break

            self.find_best_splits()

            best_leaf = self.get_max_gain()
            self.split(new_tree, best_leaf, left_leaf, right_leaf)
        return

    def get_max_gain(self):
        best_leaf = None
        current_gain = 0.0
        for leaf, split_info in self._best_split_per_leaf.iteritems():
            if split_info.gain > current_gain:
                best_leaf = leaf
                current_gain = split_info.gain

        return best_leaf

    def before_train(self):
        # TODO: 利用gradients和hessians做一些事情
        self._histogram_pool.resetMap()
        return

    def find_best_splits(self):
        # 根据当前的feature发现最佳的切割点
        is_feature_used = [True for x in xrange(self._num_features)]

        self.construct_histograms(is_feature_used)
        self.find_best_split_from_histograms(is_feature_used)
        return

    def construct_histograms(self, is_feature_used):
        # construct smaller leaf
        self._smaller_leaf_histogram_array = self._train_data.construct_histograms(
            is_feature_used,
            self._smaller_leaf_split.data_indices(),
            self._smaller_leaf_split.leaf_index(),
            self._gradients,
            self._hessians,
        )

        # construct larger leaf
        self._larger_leaf_histogram_array_ = self._train_data.construct_histograms(
            is_feature_used,
            self._larger_leaf_split.data_indices(),
            self._larger_leaf_split.leaf_index(),
            self._gradients,
            self._hessians,
        )
        return

    def find_best_split_from_histograms(self, is_feature_used):

        smaller_best = SplitInfo()
        larger_best = SplitInfo()

        for feature_index in xrange(self._num_features):
            if not is_feature_used(feature_index):
                continue

            # self._train_data.fix_histograms()
            smaller_split = self._smaller_leaf_histogram_array[feature_index].find_best_threshold(
                self._smaller_leaf_split.sum_gradients(),
                self._smaller_leaf_split.sum_hessians(),
                self._smaller_leaf_split.num_data_in_leaf(),
            )

            if smaller_split > smaller_best:
                smaller_best = smaller_split

            larger_split = self._larger_leaf_histogram_array_[feature_index].find_best_threshold(
                self._larger_leaf_split.sum_gradients(),
                self._larger_leaf_split.sum_hessians(),
                self._larger_leaf_split.num_data_in_leaf(),
            )
            if larger_split > larger_best:
                larger_best = larger_split

        leaf = self._smaller_leaf_split.leaf_index()
        self._best_split_per_leaf[leaf] = smaller_best

        leaf = self._larger_leaf_split.leaf_index()
        self._best_split_per_leaf[leaf] = larger_best

        return

    def before_find_best_leave(self, new_tree, left_leaf, right_leaf):
        return True

    def split(self, new_tree, best_leaf, left_leaf, right_leaf):
        best_split_info = self._best_split_per_leaf[best_leaf]

        default_value = 0.0

        right_leaf = new_tree.split(
            best_leaf,
            best_split_info,
        )

        # TODO: data_partition

        # init the leaves that used on next iteration: smaller_leaf_splits_ used for split

        return


if __name__ == '__main__':
    pass
