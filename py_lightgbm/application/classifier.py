# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: classifier.py
@time: 2017/7/9 12:03
@change_time:
1.2017/7/9 12:03
"""
from py_lightgbm.boosting import boosting
from py_lightgbm.io.dataset import Dataset


class LGBMClassifier(object):

    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=10, max_bin=255,
                 subsample_for_bin=50000, objective=None, min_split_gain=0,
                 min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                 colsample_bytree=1, reg_alpha=0, reg_lambda=0,
                 seed=0, nthread=-1, silent=True, **kwargs):

        self._boosting_type = boosting_type     # 仅支持gdbt
        self._boosting = boosting.Booster()
        self._train_data = None

        self._num_leaves = num_leaves
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._max_bin = max_bin
        self._min_split_gain = min_split_gain
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._silent = silent
        self._min_child_samples = min_child_samples

        self._bin_mappers = None
        return

    def fit(self, X, y, sample_weight=None, init_score=None,
            group=None, eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_group=None, eval_metric=None, early_stopping_rounds=None,
            verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None):
        self._train_data = Dataset(X, y, feature_name)
        self._bin_mappers = self._train_data.create_bin_mapper(self._max_bin)
        self._train_data.construct(self._bin_mappers)
        self._boosting.init(self._train_data, num_leaves=self._num_leaves)

        for i in xrange(self._n_estimators):
            self._boosting.train_one_iter(self._train_data)
        return

    def show(self):
        self._boosting.show()
        return

    def print_bin_mappers(self):
        for bin_mapper in self._bin_mappers:
            print ""
            print "bins", bin_mapper._num_bins
            print "bin_upper_bound", bin_mapper._bin_upper_bound
            print "min", bin_mapper._min_value
            print "max", bin_mapper._max_value
            print ""
        return

    def predict(self, X, raw_score=False, num_iteration=0):

        return

    def score(self, X, y):

        return
