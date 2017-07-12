# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: split_info.py
@time: 2017/7/9 22:04
@change_time:
1.2017/7/9 22:04
"""


class SplitInfo(object):
    def __init__(self):
        self.feature_index = -1
        self.threshold_bin = -1

        self.left_output = 0
        self.right_output = 0
        self.gain = 0

        self.left_count = 0
        self.right_count = 0

        self.left_sum_gradients = 0
        self.left_sum_hessians = 0
        self.right_sum_gradients = 0
        self.right_sum_hessians = 0
        return

    def __str__(self):
        repr_str = "index:{0};bin:{1};gain:{2};lc:{3};rc:{4}".format(
            self.feature_index,
            self.threshold_bin,
            self.gain,
            self.left_count,
            self.right_count,
        )
        return repr_str

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.feature_index = -1
        self.threshold_bin = -1

        self.left_output = 0
        self.right_output = 0
        self.gain = 0

        self.left_count = 0
        self.right_count = 0

        self.left_sum_gradients = 0
        self.left_sum_hessians = 0
        self.right_sum_gradients = 0
        self.right_sum_hessians = 0
        return


if __name__ == '__main__':
    pass
