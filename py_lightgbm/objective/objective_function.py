# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: objective_function.py
@time: 2017/7/9 17:41
@change_time:
1.2017/7/9 17:41
"""
import numpy as np


class BinaryObjective(object):
    def __init__(self, labels):
        self._sigmoid = 1.0
        self._labels = labels
        return

    def get_gradients(self, scores):
        gradients = np.zeros(scores.shape)
        hessians = np.zeros(scores.shape)

        for i in xrange(self._labels.shape[0]):
            label = self._labels[i]
            response = -label * self._sigmoid / (1 + np.exp(label * self._sigmoid * scores[i]))
            abs_response = np.abs(response)
            gradients[i] = response
            hessians[i] = abs_response * (self._sigmoid - abs_response)

        return gradients, hessians


if __name__ == '__main__':
    labels = np.array([[1], [1], [-1], [1], [-1]])
    scores = np.array([[0.5], [-1], [-1], [1], [0.5]])
    bo = BinaryObjective(labels)
    gradients, hessians = bo.get_gradients(scores)
    print gradients
    print hessians