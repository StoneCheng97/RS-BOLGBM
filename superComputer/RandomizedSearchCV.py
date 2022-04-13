import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Create the dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
test_data = raw_data.dropna()
# 全特征
train_features, train_labels = test_data.drop(labels=['Status'], axis=1).values, test_data['Status'].values
# SelectKBest选择的特征
# train_features, train_labels = test_data.drop(labels=['Status', 'RNP', 'User_ID', 'Queue_Number'], axis=1).values, \
#                                test_data['Status'].values
# ReliefF选择的特征
# train_features, train_labels = test_data.drop(labels=['id', 'Submit_Time', 'Run_Time', 'Status'], axis=1).values, \
#                                test_data['Status'].values
# 参数定义区
lgb_param = dict(class_weight=[None, 'balanced'], boosting_type=['gbdt', 'goss', 'dart'],
                 num_leaves=list(range(30, 150)),
                 learning_rate=list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
                 subsample_for_bin=list(range(20000, 300000, 20000)),
                 min_child_samples=list(range(20, 500, 5)),
                 n_estimators=sp_randint(100, 500)
                 # reg_lambda=list(np.linspace(0, 1)),
                 # reg_alpha=list(np.linspace(0, 1)),
                 # colsample_bytree=list(np.linspace(0.6, 1, 10))
                 )
# knn_param = dict(
#     n_neighbors=list(range(3, 30))
# )
knn_param = dict(
    n_neighbors=sp_randint(3, 30)
)
DT_param = dict(
    criterion=['gini', 'entropy'],
    splitter=['best', 'random'],
    max_depth=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

)
RF_param = dict(
    criterion=['gini', 'entropy'],
    n_estimators=list(range(100, 500)),
    bootstrap=[True, False],
    max_depth=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

)
SVM_param = dict(
    kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    C=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    gamma=['scale', 'auto']

)
GDBT_param = dict(
    criterion=['friedman_mse', 'squared_error'],
    learning_rate=list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
    n_estimators=list(range(100, 500))
)


def KNN(params):
    knn_random = KNeighborsClassifier()
    clf = RandomizedSearchCV(knn_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    #     SKB
    # {'n_neighbors': 27}


def DT(params):
    dt_random = DecisionTreeClassifier()
    clf = RandomizedSearchCV(dt_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    #     SKB
    # {'n_neighbors': 27}


def RF(params):
    rf_random = RandomForestClassifier()
    clf = RandomizedSearchCV(rf_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    #     SKB
    # {'n_neighbors': 27}


def SVM(params):
    rf_random = RandomForestClassifier()
    clf = RandomizedSearchCV(rf_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    #     SKB
    # {'n_neighbors': 27}


def GDBT(params):
    gdbt_random = GradientBoostingClassifier()
    clf = RandomizedSearchCV(gdbt_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    #     SKB
    # {'n_neighbors': 27}


def lightGBM(params):
    lgb_random = lgb.LGBMClassifier()
    clf = RandomizedSearchCV(lgb_random, params, random_state=0, n_jobs=-1, verbose=2, cv=10, n_iter=20)
    search = clf.fit(train_features, train_labels)
    print("Best: ", search.best_params_)
    # SKB
    # {'subsample_for_bin': 80000, 'num_leaves': 75, 'min_child_samples': 265, 'learning_rate': 0.050079054996995706, 'class_weight': None, 'boosting_type': 'dart'}
    # {'num_leaves': 53, 'learning_rate': 0.19926284742685216, 'class_weight': None, 'boosting_type': 'goss'}
    # {'subsample_for_bin': 280000, 'num_leaves': 31, 'min_child_samples': 330, 'learning_rate': 0.12605847206065268, 'class_weight': None, 'boosting_type': 'goss'}


if __name__ == '__main__':
    KNN(knn_param)
    DT(DT_param)
    RF(RF_param)
    # GDBT(GDBT_param)
    # lightGBM(lgb_param)
