# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: utils.py
@time: 2017/7/13 20:48
@change_time:
1.2017/7/13 20:48
"""
import datetime
import time


def dec_timer(func):
    def _wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        start = time.time()
        ret = func(*args, **kwargs)

        end_time = datetime.datetime.now()
        print '_'*70
        print'{} takes {:2f}s, from {} to {}'.format(func.__name__, time.time() - start, start_time, end_time)
        print '_'*70
        # print 'end at', datetime.datetime.now()
        return ret

    return _wrapper