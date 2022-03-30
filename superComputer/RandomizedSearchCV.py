import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Create the dataset
raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
test_data = raw_data.dropna()
train_features, train_labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values

# 参数定义区
lgb_param = dict(class_weight=[None, 'balanced'], boosting_type=['gbdt', 'goss', 'dart'],
                 num_leaves=list(range(30, 150)),
                 learning_rate=list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
                 subsample_for_bin=list(range(20000, 300000, 20000)),
                 min_child_samples=list(range(20, 500, 5)),
                 # n_estimators=np.array([(100,500) for i in range(500)])
                 # reg_lambda=list(np.linspace(0, 1)),
                 # reg_alpha=list(np.linspace(0, 1)),
                 # colsample_bytree=list(np.linspace(0.6, 1, 10))
                 )


def lightGBM(params):
    lgb_random = lgb.LGBMClassifier()
    clf = RandomizedSearchCV(lgb_random, params, random_state=0)
    search = clf.fit(train_features, train_labels)
    print(search.best_params_)
    # {'num_leaves': 53, 'learning_rate': 0.19926284742685216, 'class_weight': None, 'boosting_type': 'goss'}
    # {'subsample_for_bin': 280000, 'num_leaves': 31, 'min_child_samples': 330, 'learning_rate': 0.12605847206065268, 'class_weight': None, 'boosting_type': 'goss'}


if __name__ == '__main__':
    lightGBM(lgb_param)
