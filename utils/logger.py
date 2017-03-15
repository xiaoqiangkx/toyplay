# -*- coding: utf-8 -*-
#!/usr/bin/env python
# encoding: utf-8


"""
@version: 1.0
@author: xiaoqiangkx
@file: logger.py
@time: 2017/3/15 23:22
@change_time:
1.2017/3/15 23:22
"""

import logging
logging.basicConfig()

_LOGGER = logging.getLogger("toyplay")

LEVEL_NORMAL = logging.WARNING
LEVEL_DEBUG = logging.INFO

_LOGGER.setLevel(LEVEL_DEBUG)


def info(msg):
    _LOGGER.info(msg)
    return


def error(msg):
    _LOGGER.error(msg)
    return


def setLevel(level):
    _LOGGER.setLevel(level)