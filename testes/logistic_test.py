# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: logistic_test.py
@time: 2017/3/13 21:54
@change_time:
1.2017/3/13 21:54
"""
import unittest
from utils import formula
import numpy as np


class TestCase(unittest.TestCase):
    def test_sigmoid(self):
        result = formula.sigmoid(np.array([[0, 0, 0]]).T, np.array([[0, 0, 0]]).T)
        self.assertAlmostEqual(result[[0]], 0.5)
        return


if __name__ == '__main__':
    unittest.main()
