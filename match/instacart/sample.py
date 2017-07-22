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
import numpy as np

random.seed(const.RANDOM_STATE)


def sample_train(from_path, sample_path, key, percentage=0.1):
    """
    sample a dataset according to keys
    """
    from_df = pd.read_csv(from_path)
    order_id_list = list(from_df[key].value_counts().index)
    sample_order_id_set = set(random.sample(order_id_list, int(len(order_id_list) * percentage)))

    order_id_data_list = from_df[key].values
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
    target_df = pd.merge(a_df, b_df, on=key, how='left')
    target_df.to_csv(save_path, index=False)
    return


def read_sample_train(from_path, nrows=None):
    return pd.read_csv(from_path, nrows=nrows)


def make_user_product_list():
    prior_order = pd.read_csv(const.EXTEND_PRIOR_PATH)
    prior_user_product_list_df = prior_order[prior_order.eval_set == 'prior'].groupby(const.UID)[const.PID].apply(
        set).reset_index().set_index(const.UID)

    del prior_order

    train_df = pd.read_csv(const.EXTEND_TRAIN_PATH)
    train_user_product_list_df = train_df[train_df.eval_set == 'train'].groupby([const.OID, const.UID])[
        const.PID].apply(
        set).reset_index().set_index([const.OID, const.UID])

    del train_df

    order_user_product_list = {}
    cnt = 0
    for order_user_id, train_user_product_list in train_user_product_list_df.iterrows():
        order_id, user_id = order_user_id
        if cnt % 10000 == 1:
            print cnt
        prior_order_product_list = prior_user_product_list_df.ix[user_id][const.PID]

        not_in_clude_in_train_product_list = prior_order_product_list - train_user_product_list[const.PID]
        order_user_product_list[order_user_id] = not_in_clude_in_train_product_list

        cnt += 1

    total_len = sum([len(value) for value in order_user_product_list.itervalues()])
    output_data = np.zeros((total_len, 2))
    cnt = 0
    for order_user_id, product_list in order_user_product_list.iteritems():
        order_id, user_id = order_user_id
        for product_id in product_list:
            output_data[cnt, 0] = order_id
            output_data[cnt, 2] = product_id
            cnt += 1

    new_dataframe = pd.DataFrame(output_data, columns=[const.OID, const.UID, const.PID], dtype="int")
    new_dataframe.to_csv(const.NEGATIVE_TRAIN_DATA, index=False)
    return


def merge_train_negative_data(train_path, negative_path, total_train):
    train_df = pd.read_csv(train_path)
    train_df[const.LABEL] = 1
    negative_df = pd.read_csv(negative_path)
    negative_df[const.LABEL] = 0
    final_train_data = pd.concat([train_df, negative_df])
    final_train_data.to_csv(total_train, index=False)
    pass


if __name__ == '__main__':
    # extend_data(const.PRIOR_ORDERS_PATH, const.ORDERS_PATH, const.EXTEND_PRIOR_PATH, const.OID)
    # extend_data(const.TRAIN_ORDERS_PATH, const.ORDERS_PATH, const.EXTEND_TRAIN_PATH, const.OID)
    #
    # # 构造用户过去购买过的所有物品,构造反例数据,生成sample_train_failure_path
    # make_user_product_list()
    # extend_data(const.NEGATIVE_TRAIN_DATA, const.ORDERS_PATH, const.EXTEND_NEGATIVE_TRAIN_PATH, [const.OID, const.UID])

    # merge train and negative data
    # merge_train_negative_data(const.EXTEND_TRAIN_PATH, const.EXTEND_NEGATIVE_TRAIN_PATH, const.TOTAL_TRAIN_DATA)

    raw_train_orders_path = const.TOTAL_TRAIN_DATA
    sample_path = const.SAMPLE_TRAIN_PATH
    sample_train(raw_train_orders_path, sample_path, const.OID, percentage=0.1)
