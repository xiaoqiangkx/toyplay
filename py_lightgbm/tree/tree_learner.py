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


class TreeLearner(object):
    def __init__(self):
        self._gradients = None
        self._hessians = None
        self._tree_config = None        # TODO: 用于管理tree config文件
        self._best_split_per_leaf = None
        self._histogram_pool = None
        self._train_data = None
        self._max_cache_size = None

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

        new_tree = Tree()

        left_leaf = 0
        right_leaf = -1
        cur_depth = 1

        for split in xrange(self._tree_config.num_leaves - 1):
            if self.before_find_best_leave(new_tree, left_leaf, right_leaf):
                self.find_best_thresholds()
                self.find_best_splits_for_leaves()

            best_leaf = self.best_split_per_leaf_

            # TODO analyse gain information

            self.split(new_tree, best_leaf, left_leaf, right_leaf)
        return

    def before_train(self):
        # TODO: 利用gradients和hessians做一些事情
        self._histogram_pool.resetMap()

        return

    def find_best_thresholds(self):
        # TODO 根据当前的feature发现最佳的切割点
        return

    def before_find_best_leave(self):
        return

    def find_best_splits_for_leaves(self):
        return

    def split(self):
        return


if __name__ == '__main__':
    pass
