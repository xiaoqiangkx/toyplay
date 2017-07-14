#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: profile.py
@time: 2017/7/14 15:58
@change_time: 
1.2017/7/14 15:58
"""
PROFILE = None
import time


def enable_profile():
	global PROFILE
	import cProfile
	PROFILE = cProfile.Profile()
	PROFILE.enable()
	return


def close_profile():
	PROFILE.disable()
	PROFILE.dump_stats('profile_%d.prof' % time.time())
	return