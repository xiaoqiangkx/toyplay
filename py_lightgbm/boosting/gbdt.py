# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: gbdt.py
@time: 2017/7/1 11:25
@change_time:
1.2017/7/1 11:25
"""
from py_lightgbm.tree.tree_learner import TreeLearner
from py_lightgbm.objective import BinaryObjective


class Gbdt(object):
    def __init__(self):
        self._train_data = None
        self._scores = None
        self._object_function = None
        self._gradients = None
        self._hessians = None

        self._scores = None
        self._num_leaves = 0

        self._models = []
        self._coefs = []
        return

    def init(self, train_data, num_leaves):
        self._train_data = train_data
        self._scores = train_data.init_score
        self._num_leaves = num_leaves
        self._object_function = BinaryObjective(self._train_data.labels)
        return

    def train_one_iter(self, gradients=None, hessians=None):
        """
        What's the value of gradient and hessian
        """

        if gradients is None or hessians is None:
            self._boosting()
            gradients = self._gradients
            hessians = self._hessians

        # only use one tree in one iterations
        tree = TreeLearner(self._num_leaves, self._train_data)
        tree.train(gradients, hessians)
        return

    def _boosting(self):
        if not self._object_function:
            return

        self._gradients, self._hessians = self._object_function.get_gradients(self._scores)
        return


if __name__ == '__main__':
    pass
