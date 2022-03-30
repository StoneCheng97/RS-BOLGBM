# Pandas and numpy for data manipulation
import random

import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold

MAX_EVALS = 500
N_FOLDS = 10
param_grid = {
    # 'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'verbose': [-1, -1]
}
# Subsampling (only applicable with 'goss')
subsample_dist = list(np.linspace(0.5, 1, 100))

np.set_printoptions(suppress=False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 小数点后面保留5位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
test_data = raw_data.dropna()
features, labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values
train_data = lgb.Dataset(features, label=labels)
# KFold 10倍交叉验证
kf = KFold(n_splits=10)
print(kf.get_n_splits())
print(kf)
for train_index, test_index in kf.split(features):
    print("train:", train_index, "test:", test_index)
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
print("训练集：",X_train)
print("训练集labels：",y_train)
print("=================================================================")

# lgbCV
# params采样结果
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
print(params)
params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
print(params)

cv_result = lgb.cv(params, train_data, num_boost_round=1000, nfold=10, metrics='auc', early_stopping_rounds=100,
                   verbose_eval=False, seed=50)
# 最高分
cv_results_best = np.max(cv_result['auc-mean'])
# 最高分的标准差
cv_results_std = cv_result['auc-stdv'][np.argmax(['auc-mean'])]


print("the maximum ROC AUC on the validation set was{:.5f} with std of {:.5f}.".format(cv_results_best, cv_results_std))
print('The ideal number of iteractions was {}.'.format(np.argmax(cv_result['auc-mean']) + 1))
print('best n_estimator is {}'.format(len(cv_result['auc-mean'])))
print('best cv score is {}'.format(pd.Series(cv_result['auc-mean']).max()))
# {'boosting_type': 'gbdt', 'num_leaves': 97, 'learning_rate': 0.007206637980435866, 'subsample_for_bin': 80000, 'min_child_samples': 65, 'reg_alpha': 0.4081632653061224, 'reg_lambda': 0.26530612244897955, 'colsample_bytree': 0.8666666666666667, 'verbose': -1}
# {'boosting_type': 'gbdt', 'num_leaves': 97, 'learning_rate': 0.007206637980435866, 'subsample_for_bin': 80000, 'min_child_samples': 65, 'reg_alpha': 0.4081632653061224, 'reg_lambda': 0.26530612244897955, 'colsample_bytree': 0.8666666666666667, 'verbose': -1, 'subsample': 0.6767676767676768}
# the maximum ROC AUC on the validation set was0.97979 with std of 0.00184.
# The ideal number of iteractions was 1000.
# best n_estimator is 1000
# best cv score is 0.9797873428836741

# {'boosting_type': 'goss', 'num_leaves': 89, 'learning_rate': 0.041025916113156854, 'subsample_for_bin': 280000, 'min_child_samples': 365, 'reg_alpha': 0.4693877551020408, 'reg_lambda': 0.9387755102040816, 'colsample_bytree': 0.6888888888888889, 'verbose': -1}
# {'boosting_type': 'goss', 'num_leaves': 89, 'learning_rate': 0.041025916113156854, 'subsample_for_bin': 280000, 'min_child_samples': 365, 'reg_alpha': 0.4693877551020408, 'reg_lambda': 0.9387755102040816, 'colsample_bytree': 0.6888888888888889, 'verbose': -1, 'subsample': 1.0}
# the maximum ROC AUC on the validation set was0.97821 with std of 0.00202.
# The ideal number of iteractions was 1000.
# best n_estimator is 1000
# best cv score is 0.9782078814290006
