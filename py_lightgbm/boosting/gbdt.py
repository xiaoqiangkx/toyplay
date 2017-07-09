# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: gbdt.py
@time: 2017/7/1 11:25
@change_time:
1.2017/7/1 11:25
"""
from py_lightgbm.tree.tree import Tree


class Gbdt(object):
    def __init__(self):

        self._models = []                   # ??记录所有models
        self._gradients = []
        self._hessian = []
        self._num_tree_per_iteration = 0
        self._tree_learner = None
        self._shrinkage_rate = 0

        self._objective_function = None     # ??用于
        pass

    def train_one_iter(self, gradient, hessian, is_eval):
        """
        What's the value of gradient and hessian
        """
        if not self._models:
            new_tree = Tree()
            # TODO initial new_tree
            self._models.append(new_tree)

        if gradient is None or hessian is None:
            self.boosting()
            gradient = self._gradients
            hessian = self._hessian

        should_continue = False
        for i in xrange(self._num_tree_per_iteration):
            new_tree = self._tree_learner.train(gradient, hessian, i)

            if new_tree.num_leaves() > 0:
                should_continue = True
                new_tree.shrinkage(self._shrinkage_rate)
            else:
                if len(self._models) < self._num_tree_per_iteration:
                    new_tree.split()

            self._models.append(new_tree)

            if not should_continue:
                for i in xrange(self._num_tree_per_iteration):
                    self._models.pop()
                return True

        return

    def boosting(self):
        # TODO 使用objective_function生成gradient和hessian
        return


if __name__ == '__main__':
    pass
