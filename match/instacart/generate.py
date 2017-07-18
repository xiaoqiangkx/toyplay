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


@dec_timer
def ratio(df):
    return df


def feature_creation(df):
    """
    构造特征数据:
    """
    # user-product feature
    # user-order feature
    # product-order feature

    # ratio
    df = ratio(df)

    # diff

    # rank

    # rule

    return df


if __name__ == '__main__':
    raw_train_df = sample.read_sample_train(const.SAMPLE_TRAIN_PATH)
    train_feature_df = feature_creation(raw_train_df)
    train_feature_df.to_csv(const.SAMPLE_TRAIN_FC_PATH)
