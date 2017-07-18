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


def create_model(train_df, train_labels, validate_df, validate_labels):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 96,
        'max_depth': 10,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'learning_rate': 0.1,
        'min_child_samples': 50,
        'reg_lambda': 0.7,
        'n_estimators': 400,
        'silent': True
    }

    print('light GBM train :-)')
    clf = lgb.LGBMClassifier(**params)
    result = clf.fit(
        train_df,
        train_labels,
        eval_set=[(validate_df, validate_labels), ],
        eval_metric=['auc', ],
        categorical_feature=['aisle_id', 'department_id'],
        early_stopping_rounds=10,
    )
    return result

if __name__ == '__main__':
    train_df = None
    validate_df = None
    train_labels = None
    validate_labels = None
    model = create_model(train_df, train_labels, validate_df, validate_labels)
