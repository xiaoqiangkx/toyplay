# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: tunning.py
@time: 2017/7/14 07:16
@contact: bywangqiang@foxmail.com
@change_time:
1.2017/7/14 07:16
2.2017/7/18 22:11 使用LightGBM的坐标下降法来进行调参
"""
import lightgbm as lgb
import pandas as pd
import const
import numpy as np

import random
random.seed(const.RANDOM_STATE)
from sklearn import metrics


def split_train_test(data, key, percentage=0.2):
    """
    根据key值来划分数据为train和test数据
    """
    key_id_list = list(data[key].value_counts().index)
    sample_key_set = set(random.sample(key_id_list, int(len(key_id_list) * percentage)))

    key_id_data_list = data[key].values
    sample_index_flag = range(data.shape[0])
    for idx, value in enumerate(key_id_data_list):
        if value in sample_key_set:
            sample_index_flag[idx] = True
        else:
            sample_index_flag[idx] = False

    test_df = data[sample_index_flag]
    train_df = data[[not x for x in sample_index_flag]]
    return train_df, test_df


def training_model(train_data, key):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 72,
        'max_depth': 10,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'learning_rate': 0.1,
        'min_child_samples': 50,
        'reg_lambda': 0.7,
        'n_estimators': 50,
        'silent': True,
        'metric': ['auc', ]
    }

    train_df = train_data.drop([key], axis=1)
    train_labels = train_data[key]
    print('light GBM train :-)')
    clf_list = []
    clf = lgb.LGBMClassifier(**params)
    categorical_feature = ['order_dow', 'order_hour_of_day', ]
    clf.fit(
        train_df,
        train_labels,
        categorical_feature=categorical_feature,
    )
    clf_list.append(clf)
    return clf_list


def mean_f1_score(order_id_index, predict_y):
    f_score_total = 0
    n = 0
    idx = 0
    for _, value in order_id_index.iterrows():
        label_list = value['label']
        predict_label_list = predict_y[idx:idx + len(label_list)]
        if not np.any(label_list) and not np.any(predict_label_list):
            f_score = 1
        else:
            f_score = metrics.f1_score(predict_label_list, label_list)
        f_score_total += f_score
        idx += len(label_list)
        n += 1

    return f_score_total / n


def cal_score(clf, train_validate_data, key):
    validate_df = train_validate_data.drop([key], axis=1)
    validate_labels = train_validate_data['label']
    order_id_index = n_test.groupby(const.OID)[key].apply(list).reset_index()
    predict_y = clf.predict_proba(validate_df)[:, 1]

    best_score = 0
    best_margin = 0

    for margin in range(1, 100):
        threshold = margin / float(100)
        predict_result = predict_y >= threshold
        # score = metrics.f1_score(validate_labels.values, predict_result)
        score = mean_f1_score(order_id_index, predict_result)
        if score > best_score:
            best_score = score
            best_margin = threshold

    return best_score, best_margin


def get_sample_negative_data(data, sample_percentage):
    order_id_index = data.groupby(const.OID)[const.PID].apply(list).reset_index()
    result = data.set_index([const.OID, const.PID])
    sample_index_list = []
    for order_id, value in order_id_index.iterrows():
        order_id = value['order_id']
        product_id_list = value['product_id']
        num = int(len(product_id_list) * sample_percentage)
        if num <= 0:
            continue

        product_sample = random.sample(product_id_list, num)
        sample_index_list.extend([(order_id, product_id) for product_id in product_sample])

    new_data = result.ix[sample_index_list]
    new_data.reset_index(inplace=True)
    return new_data


if __name__ == '__main__':
    total_train = pd.read_csv(const.SAMPLE_TRAIN_FC_PATH)
    num_data = total_train.shape[0]

    n_train, n_test = split_train_test(total_train, const.OID, 0.2)
    negative_data = n_train[n_train.label == 0]
    positive_data = n_train[n_train.label == 1]

    num_positive = positive_data.shape[0]
    num_negative = negative_data.shape[0]

    target_num = 2 * num_positive
    sample_negative_data = get_sample_negative_data(negative_data, target_num / float(num_negative))
    train_data = pd.concat([sample_negative_data, positive_data])

    drop_list = ['order_id', 'product_id', 'reordered', 'add_to_cart_order', 'user_id', 'eval_set', 'order_number']
    train_data.drop(drop_list, axis=1, inplace=True)
    train_validate_data = n_test.drop(drop_list, axis=1)

    clf_list = training_model(train_data, 'label')

    for clf in clf_list:
        result = cal_score(clf, train_validate_data, 'label')
        print "result", result
