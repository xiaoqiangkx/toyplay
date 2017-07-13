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
import numpy as np


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
        self._tree_list = []        # 记录所有的树
        self._tree_coef = []        # 记录所有的因子
        return

    def init(self, train_data, num_leaves):
        self._train_data = train_data
        self._scores = train_data.init_score
        self._num_leaves = num_leaves
        self._object_function = BinaryObjective(self._train_data.labels)
        return

    def train_one_iter(self, train_data, learning_rate, gradients=None, hessians=None):
        """
        What's the value of gradient and hessian
        """

        if gradients is None or hessians is None:
            self._boosting()
            gradients = self._gradients
            hessians = self._hessians

        # only use one tree in one iterations
        tree_learner = TreeLearner(self._num_leaves, self._train_data)
        tree = tree_learner.train(gradients, hessians)
        self._tree_list.append(tree)

        self._update_scores(tree, tree_learner, learning_rate)
        return

    def _update_scores(self, tree, tree_learner, learning_rate):
        """
        更新循环结束以后所有值的score
        """
        for i in xrange(self._num_leaves):
            output = tree.output_of_leaf(i)

            indices = tree_learner.get_indices_of_leaf(i)
            for index in indices:
                self._scores[index] += learning_rate * output
        return

    def _boosting(self):
        if not self._object_function:
            return

        self._gradients, self._hessians = self._object_function.get_gradients(self._scores)
        return

    def predict_proba(self, X):
        """
        predict result according to tree_list
        """
        result = np.zeros((X.shape[0], ))
        for idx, tree in enumerate(self._tree_list):
            predict_y = tree.predict_prob(X)
            result += predict_y

        return self._object_function.convert_output(result)

    def show(self):
        """
        展示所有的tree信息
        """
        for index, tree in enumerate(self._tree_list):
            print "\n======================show tree {0}".format(index + 1)
            tree.show()
        return

if __name__ == '__main__':
    pass
