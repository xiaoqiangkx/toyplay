# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: generate.py
@time: 2017/7/14 07:16
@contact: bywangqiang@foxmail.com
@change_time:
1.2017/7/14 07:16 初始化
2.2017/7/18 15:48 构造feature_engineering
"""
import sample
import const
from utils import dec_timer
import pandas as pd
import numpy as np


def load_prior_data():
    df = pd.read_csv(const.EXTEND_PRIOR_PATH)
    return df


@dec_timer
def ratio(df):

    return df


@dec_timer
def diff(df):
    # 距离上一次购买这个商品过了多久时间
    df['order_number-user_product_order_number_max'] = df['order_number'] - df['user_product_order_number_max']

    # TODO 可以关心距离这个aisle和department过了多久时间
    return df


@dec_timer
def build_field_feature(prior, group_id):
    # TODO: order_dow, order_hour_of_day如何处理??
    field_key = {const.PID: "product", const.UID: "user"}[group_id]
    df = prior.groupby(group_id).agg({
        const.UID: {"%s_bought_time" % field_key: np.size},
        const.PID: {"%s_item_unique_number" % field_key: pd.Series.nunique},
        const.OID: {"%s_order_number" % field_key: pd.Series.nunique},
        const.ORDER_NUM: {"%s_order_number_max" % field_key: np.max},
        const.ADD_TO_CART_NUM: {
            "%s_item_add_to_cart_mean" % field_key: np.mean,
        },
        const.REORDER: {
            "%s_reordered_times" % field_key: np.sum,
            "%s_reordered_ratio" % field_key: np.mean,
        },
        const.DAYS_SINCE_PRIOR_ORDER: {
            "%s_days_prior_order_mean" % field_key: np.mean,
            "%s_active_days" % field_key: np.sum,       # 表征用户活跃的整个时间跨度
        }

    }).reset_index()

    df.columns = [group_id] + list(df.columns.droplevel(0))[1:]
    return df


@dec_timer
def  build_interactive_feature(prior_data, key_1, key_2):

    group_id = [key_1, key_2]
    field_key = {
        (const.UID, const.PID): "user_product",
    }[(key_1, key_2)]

    df = prior_data.groupby(group_id).agg({
        const.OID: {"%s_bought_times" % field_key: pd.Series.nunique},
        const.ORDER_NUM: {
            "%s_order_number_max" % field_key: np.max,
            "%s_order_number_min" % field_key: np.min,
        },
        const.ADD_TO_CART_NUM: {
            "%s_item_add_to_cart_mean" % field_key: np.mean,
            "%s_item_add_to_cart_min" % field_key: np.min,
            "%s_item_add_to_cart_max" % field_key: np.max,
        },
        const.REORDER: {
            "%s_reordered_times" % field_key: np.sum,
            "%s_reordered_ratio" % field_key: np.mean,
        },
        const.DAYS_SINCE_PRIOR_ORDER: {
            "%s_days_prior_order_mean" % field_key: np.mean,
            "%s_days_prior_order_min" % field_key: np.min,
            "%s_days_prior_order_max" % field_key: np.max,
        }

    }).reset_index()

    df.columns = group_id + list(df.columns.droplevel(0))[2:]
    return df


def feature_creation(df):
    """
    构造特征数据:
    """
    prior_data = load_prior_data()

    # ------------ 1. field feature -----------------------
    # 1.1 user feature
    user_feature = build_field_feature(prior_data, const.UID)
    df = df.merge(user_feature, how='left', on=[const.UID])

    # 1.2 product feature
    product_feature = build_field_feature(prior_data, const.PID)
    df = df.merge(product_feature, how='left', on=[const.PID])

    # 1.3 order 包括df中order_dow, order_hour_of_day等字段, 已包含

    # ------------ 2. interactive feature -----------------
    # 2.1 user-product feature
    user_product_feature = build_interactive_feature(prior_data, const.UID, const.PID)
    user_product_feature.set_index([const.UID, const.PID])
    df.set_index([const.UID, const.PID])

    df = df.merge(user_product_feature, how='left', on=[const.UID, const.PID])

    # 2.2 user-order feature

    # 2.3 product-order feature

    # ------------ 3. experience feature ------------------

    # ratio
    df = ratio(df)

    # diff
    df = diff(df)

    # rank

    # rule

    del prior_data      # 删除数据
    return df


if __name__ == '__main__':
    raw_train_df = sample.read_sample_train(const.SAMPLE_TRAIN_PATH)
    raw_train_df.set_index([const.OID, const.UID, const.PID])
    train_feature_df = feature_creation(raw_train_df)
    train_feature_df.to_csv(const.SAMPLE_TRAIN_FC_PATH, index=False)
