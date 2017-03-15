# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: logistic.py
@time: 2017/3/12 23:47
@change_time:
1.2017/3/12 23:47
"""
import numpy as np
from utils import file_utils as FU
from utils import formula
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, delta=0.0001, alpha=0.1):
        """
        init hyper-parameters
        """
        self.w = None
        self.size = None
        self.delta = delta
        self.theshold = 0.5
        self.loss_data = []
        self.alpha = alpha
        return

    def init_params(self, item_size):
        self.w = np.random.rand(*item_size)
        self.size = item_size
        self.loss_data = []
        return

    def fit(self, X, Y):
        """
        Train model using training data and hyper-parameters
        """
        item_size = (X.shape[0], 1)
        self.init_params(item_size)

        num = 0
        last_loss = np.inf
        step = 0
        while True:
            self.w -= self.cal_delta(X, Y).astype(float)
            num += 1
            loss = self.cal_loss(X, Y)
            if np.abs(last_loss - loss) <= self.delta:
                break
            else:
                step += 1
                last_loss = loss
                self.loss_data.append(loss)
                # print "step:", step, "loss:", loss

        return

    def predict(self, X):
        data = formula.sigmoid(X, self.w) >= self.theshold
        y = np.zeros((1, X.shape[1]))
        for index, value in enumerate(data[0, :]):
            if value:
                y[0, index] = 1
        return y

    def cal_delta(self, X, Y):
        y = formula.sigmoid(X, self.w)
        delta_w_1 = - np.dot(X, (Y - y).T)

        # delta_w_2 = np.zeros((self.size[0], self.size[0]))
        # for i in xrange(X.shape[1]):
        #     delta_w_2 += (np.dot(X[:, i:i+1], X[:, i:i+1].T) * np.dot(y, (1 - y).T)).astype(float)
        # return np.dot(np.linalg.inv(delta_w_2), delta_w_1)
        return self.alpha * delta_w_1

    def cal_loss(self, X, Y):
        loss = 0
        for i in xrange(X.shape[1]):
            temp = np.dot(self.w.T, X[:, i])[[0]]
            loss += -Y[0, i] * temp + np.log(1 + np.exp(temp.astype(float)))
        return loss

    def draw_data(self, X, Y):
        np_index1 = [index for index, x in enumerate(Y[0]) if x == 0]
        np_index2 = [index for index, x in enumerate(Y[0]) if x == 1]

        plt.scatter(X[0, np_index1], X[1, np_index1], c='y', marker='o')
        plt.scatter(X[0, np_index2], X[1, np_index2], c='b', marker='v')
        return

    def draw_line(self, X):
        min1 = np.min(X[0, :])
        max1 = np.max(X[0, :])

        min2 = - (self.w[2] + self.w[0] * min1) / self.w[1]
        max2 = - (self.w[2] + self.w[0] * max1) / self.w[1]

        plt.plot([min1, max1], [min2, max2], 'r')
        return

    def draw_loss(self):
        plt.plot(self.loss_data)
        return


if __name__ == '__main__':
    filename = u"../data/watermelon.csv"
    X, Y = FU.load_water_melon_data(filename)
    X = formula.plus_one(X)
    model = LogisticRegression()
    model.fit(X, Y)

    # model.draw_data(X, Y)
    # model.draw_line(X)
    model.draw_loss()
    plt.show()

