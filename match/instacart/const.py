# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: const.py
@time: 2017/7/14 07:16
@contact: bywangqiang@foxmail.com
@change_time:
1.2017/7/14 07:16
"""

RANDOM_STATE = 42

SAMPLE_TRAIN_PATH = "data/sample_train.csv"
EXTEND_PRIOR_PATH = "data/extend_prior_order.csv"
EXTEND_TRAIN_PATH = "data/extend_train_order.csv"
EXTEND_NEGATIVE_TRAIN_PATH = "data/extend_negative_train.csv"

NEGATIVE_TRAIN_DATA = "data/negative_train.csv"
TOTAL_TRAIN_DATA = "data/total_train.csv"


TRAIN_ORDERS_PATH = "../../data/instacart/order_products__train.csv"
AISLES_PATH = "../../data/instacart/aisles.csv"
DEPARTMENTS = "../../data/instacart/departments.csv"
PRIOR_ORDERS_PATH = "../../data/instacart/order_products__prior.csv"
ORDERS_PATH = "../../data/instacart/orders.csv"
PRODUCTS_PATH = "../../data/instacart/products.csv"


SAMPLE_TRAIN_FC_PATH = "data/sample_train_fc.csv"


UID = 'user_id'
PID = 'product_id'
OID = 'order_id'
AID = 'aisle_id'
DID = 'department_id'
ORDER_NUM = "order_number"
ADD_TO_CART_NUM = "add_to_cart_order"
REORDER = "reordered"
ORDER_DOW = "order_dow"
ORDER_HOUR_OF_DAY = "order_hour_of_day"
DAYS_SINCE_PRIOR_ORDER = "days_since_prior_order"
LABEL = "labels"
