# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: formula.py
@time: 2017/3/13 21:55
@change_time:
1.2017/3/13 21:55
"""
import numpy as np


def sigmoid(x, w):
    return 1 / (1 + np.exp(np.dot(-w.T, x)))
