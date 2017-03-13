# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: file_utils.py
@time: 2017/3/13 22:00
@change_time:
1.2017/3/13 22:00
"""

import pandas as pd
import numpy as np


def load_csv_data(filename):
    df = pd.read_csv(filename)
    m, n = df.shape
    X = df.values[:, :n-1]
    X = np.column_stack((X, np.ones(m)))
    Y = df.values[:, n-1:n]
    return X.T, Y.T
