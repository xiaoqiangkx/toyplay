# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: sample.py
@time: 2017/7/13 20:51
@contact: bywangqiang@foxmail.com
@change_time:
1.2017/7/13 20:51 sample test data
"""
import pandas as pd
import const
import random
from utils import dec_timer

random.seed(const.RANDOM_STATE)


def sample_train(from_path, sample_path, key, percentage=0.1):
    """
    sample a dataset according to keys
    """
    from_df = pd.read_csv(from_path)
    order_id_list = list(from_df[key].value_counts().index)
    sample_order_id_set = set(random.sample(order_id_list, int(len(order_id_list) * percentage)))

    order_id_data_list = from_df['order_id'].values
    sample_index_flag = range(from_df.shape[0])
    for idx, value in enumerate(order_id_data_list):
        if value in sample_order_id_set:
            sample_index_flag[idx] = True
        else:
            sample_index_flag[idx] = False

    sample_df = from_df[sample_index_flag]
    sample_df.to_csv(sample_path, index=False)
    return


@dec_timer
def extend_data(path_a, path_b, save_path, key):
    a_df = pd.read_csv(path_a)
    b_df = pd.read_csv(path_b)
    target_df = pd.merge(a_df, b_df, on=key)
    target_df.to_csv(save_path, index=False)
    return


def read_sample_train(from_path, nrows=None):
    return pd.read_csv(from_path, nrows=nrows)


if __name__ == '__main__':
    extend_data(const.PRIOR_ORDERS_PATH, const.ORDERS_PATH, const.EXTEND_PRIOR_PATH, const.OID)
    extend_data(const.TRAIN_ORDERS_PATH, const.ORDERS_PATH, const.EXTEND_TRAIN_PATH, const.OID)

    raw_train_orders_path = const.EXTEND_TRAIN_PATH
    sample_path = const.SAMPLE_TRAIN_PATH
    sample_train(raw_train_orders_path, sample_path, const.OID, percentage=0.1)


