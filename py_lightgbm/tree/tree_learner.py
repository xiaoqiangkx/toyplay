# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: tree.py
@time: 2017/7/1 11:26
@change_time:
1.2017/7/1 11:26
"""

import copy


from py_lightgbm.tree.tree import Tree
from py_lightgbm.tree.split_info import SplitInfo
from py_lightgbm.tree.leaf_splits import LeafSplits
from py_lightgbm.tree.data_partition import DataPartition
from py_lightgbm.utils import const

from py_lightgbm.logmanager import logger
from py_lightgbm.utils import conf


_LOGGER = logger.get_logger("TreeLearner")


class TreeLearner(object):
    def __init__(self, tree_config, train_data):
        self._gradients = None
        self._hessians = None
        self._tree_config = None        # TODO: 用于管理tree config文件
        self._histogram_pool = None
        self._train_data = train_data
        self._max_cache_size = None
        self._tree_config = tree_config
        self._num_leaves = tree_config.num_leaves
        self._num_features = self._train_data.num_features
        self._num_data = self._train_data.num_data

        self._smaller_leaf_histogram_array = []
        self._larger_leaf_histogram_array = []
        self._best_split_per_leaf = None      # store all the split info

        self._smaller_leaf_split = LeafSplits(self._num_data)     # store the best splits for this leaf at smaller leaf
        self._larger_leaf_split = LeafSplits(self._num_data)
        self._data_partition = None
        self.init()
        return

    def init(self):
        # self._histogram_pool.DynamicChangeSize(self._train_data, self._tree_config,
        #                                        self._max_cache_size, self._tree_config.num_leaves)

        self._best_split_per_leaf = [SplitInfo() for _ in xrange(self._num_leaves)]
        self._data_partition = DataPartition(self._num_data, self._num_leaves)
        return

    def train(self, gradients, hessians):
        self._gradients = gradients
        self._hessians = hessians

        self.before_train()

        new_tree = Tree(self._num_leaves)

        left_leaf = 0
        right_leaf = -1
        cur_depth = 1

        # 增加重要日志信息
        for split in xrange(self._num_leaves - 1):
            # print "current split num_leave:", split

            if self.before_find_best_leave(new_tree, left_leaf, right_leaf):    # 检查数据
                self.log_before_split()
                self.find_best_splits()

            best_leaf = self.get_max_gain()
            if best_leaf is None:
                break

            self.log_split()
            left_leaf, right_leaf = self.split(new_tree, best_leaf)
            self.log_after_split()

            cur_depth = max(cur_depth, new_tree.depth_of_leaf(left_leaf))
        return new_tree

    def get_indices_of_leaf(self, idx):
        return self._data_partition.get_indices_of_leaf(idx)

    def get_max_gain(self):
        best_leaf = None
        current_gain = 0.0
        for leaf, split_info in enumerate(self._best_split_per_leaf):
            if split_info.gain > current_gain:
                best_leaf = leaf
                current_gain = split_info.gain

        return best_leaf

    def before_train(self):
        # self._histogram_pool.resetMap()

        # data_partition等元素
        self._data_partition.init()

        # 初始化smaller_leaf_split等
        for i in xrange(self._num_leaves):
            self._best_split_per_leaf[i].reset()

        self._smaller_leaf_split.init(self._gradients, self._hessians)
        self._larger_leaf_split.reset()

        return

    def find_best_splits(self):
        # 根据当前的feature发现最佳的切割点
        is_feature_used = [True] * self._num_features

        self.construct_histograms(is_feature_used)
        self.find_best_split_from_histograms(is_feature_used)
        return

    def construct_histograms(self, is_feature_used):
        # construct smaller leaf
        self._smaller_leaf_histogram_array = self._train_data.construct_histograms(
            is_feature_used,
            self._smaller_leaf_split.data_indices,
            self._smaller_leaf_split.leaf_index,
            self._gradients,
            self._hessians,
        )

        # construct larger leaf
        self._larger_leaf_histogram_array = self._train_data.construct_histograms(
            is_feature_used,
            self._larger_leaf_split.data_indices,
            self._larger_leaf_split.leaf_index,
            self._gradients,
            self._hessians,
        )
        return

    def find_best_split_from_histograms(self, is_feature_used):

        smaller_best = SplitInfo()
        larger_best = SplitInfo()

        for feature_index in xrange(self._num_features):
            if not is_feature_used[feature_index]:
                continue

            # self._train_data.fix_histograms()
            if self._smaller_leaf_histogram_array:
                smaller_split = self._smaller_leaf_histogram_array[feature_index].find_best_threshold(
                    self._smaller_leaf_split.sum_gradients,
                    self._smaller_leaf_split.sum_hessians,
                    self._smaller_leaf_split.num_data_in_leaf,
                )

                if smaller_split.gain > smaller_best.gain:
                    smaller_best = copy.deepcopy(smaller_split)

            if self._larger_leaf_histogram_array:
                larger_split = self._larger_leaf_histogram_array[feature_index].find_best_threshold(
                    self._larger_leaf_split.sum_gradients,
                    self._larger_leaf_split.sum_hessians,
                    self._larger_leaf_split.num_data_in_leaf,
                )

                if larger_split.gain > larger_best.gain:
                    larger_best = copy.deepcopy(larger_split)

        if self._smaller_leaf_split.leaf_index >= 0:
            leaf = self._smaller_leaf_split.leaf_index
            self._best_split_per_leaf[leaf] = smaller_best

        if self._larger_leaf_split.leaf_index >= 0:
            leaf = self._larger_leaf_split.leaf_index
            self._best_split_per_leaf[leaf] = larger_best
        return

    def before_find_best_leave(self, new_tree, left_leaf, right_leaf):
        # max_depth
        if new_tree.depth_of_leaf(left_leaf) >= self._tree_config.max_depth:
            self._best_split_per_leaf[left_leaf].gain = const.MIN_SCORE
            if right_leaf >= 0:
                self._best_split_per_leaf[right_leaf].gain = const.MIN_SCORE

            return False

        # min_child_samples
        if self._data_partition.counts_of_leaf(left_leaf) < self._tree_config.min_child_samples:
            self._best_split_per_leaf[left_leaf].gain = const.MIN_SCORE
            if right_leaf >= 0:
                self._best_split_per_leaf[right_leaf].gain = const.MIN_SCORE

            return False

        # TODO: histogram pool

        return True

    def split(self, new_tree, best_leaf):
        left_leaf = best_leaf
        best_split_info = self._best_split_per_leaf[best_leaf]

        right_leaf = new_tree.split(
            best_leaf,
            best_split_info,
        )

        # data_partition
        self._data_partition.split(
            left_leaf,
            self._train_data,
            best_split_info.feature_index,
            best_split_info.threshold_bin,
            right_leaf
        )

        # init the leaves that used on next iteration: smaller_leaf_splits_ used for split
        if best_split_info.left_count > best_split_info.right_count:
            self._smaller_leaf_split.init_with_data_partition(
                left_leaf,
                self._data_partition,
                self._gradients,
                self._hessians,
            )

            self._larger_leaf_split.init_with_data_partition(
                right_leaf,
                self._data_partition,
                self._gradients,
                self._hessians,
            )
        else:
            self._larger_leaf_split.init_with_data_partition(
                left_leaf,
                self._data_partition,
                self._gradients,
                self._hessians,
            )

            self._smaller_leaf_split.init_with_data_partition(
                right_leaf,
                self._data_partition,
                self._gradients,
                self._hessians,
            )
        return left_leaf, right_leaf

    def can_log(self):
        return not conf.CONFIG_DEV

    def log_before_split(self):
        """
        记录分割前情况
        """
        if not self.can_log():
            return

        # 1. 数据划分情况
        _LOGGER.info("log_before_split---------------------------------------------")
        _LOGGER.info("_data_partition:{0}".format(self._data_partition))


        # 2. 划分数据Histogram
        _LOGGER.info("smaller_leaf_split:{0}\nlarger_leaf_split:{1}\n".format(
            self._smaller_leaf_split,
            self._larger_leaf_split,
        ))
        # 3.
        _LOGGER.info("best_split:{0}".format(self._best_split_per_leaf))

        return

    def log_split(self):
        """
        记录分割情况
        """
        if not self.can_log():
            return

        _LOGGER.info("log_split---------------------------------------------")
        # 2. 划分数据Histogram
        _LOGGER.info("smaller_leaf_split:{0}\nlarger_leaf_split{1}\n".format(
            self._smaller_leaf_split,
            self._larger_leaf_split
        ))

        # 4. histogram
        _LOGGER.info("smaller leaf histogram:{0}".format(self._smaller_leaf_histogram_array))
        _LOGGER.info("larger leaf histogram:{0}".format(self._larger_leaf_histogram_array))

        _LOGGER.info("best_split:{0}".format(self._best_split_per_leaf))
        _LOGGER.info("---------------------------------------------")
        return

    def log_after_split(self):
        """
        记录分割后情况
        """
        if not self.can_log():
            return

        _LOGGER.info("log_after_split---------------------------------------------")
        _LOGGER.info("{0}".format(self._data_partition))

        _LOGGER.info("---------------------------------------------")
        return


if __name__ == '__main__':
    pass
