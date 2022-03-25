# 随机森林超参数优化- RandomSearch和GridSearch 【待定】
# 贝叶斯调优
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

MAX_EVALS = 100
N_FOLDS = 10
global ITERATION
ITERATION = 0
param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
    # 'subsample_for_bin': list(range(200000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'verbose': [-1, -1],
    "feature_pre_filter": [False],
    "force_row_wise": [True]
}
# Subsampling (only applicable with 'goss')
subsample_dist = list(np.linspace(0.5, 1, 100))
# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart',
                                 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    # 'subsample_for_bin': hp.quniform('subsample_for_bin', 200000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# Create the dataset
raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
test_data = raw_data.dropna()
train_features, train_labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values
train_set = lgb.Dataset(train_features, train_labels)

# params采样结果
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
print(params)
params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
print(params)


def objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'min_child_samples']:  # , 'subsample_for_bin'
        params[parameter_name] = int(params[parameter_name])

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=5000, early_stopping_rounds=100,
                        metrics='auc', seed=50)
    # 此部分为核心代码，

    # Extract the best score
    best_score = max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'best_score': best_score, 'estimators': n_estimators, 'status': STATUS_OK}


# Sample from the full space
example = sample(space)
# Dictionary get method with default
subsample = example['boosting_type'].get('subsample', 1.0)
# Assign top-level keys
example['boosting_type'] = example['boosting_type']['boosting_type']
print("典型样本形式:", example)

# Algorithm
tpe_algorithm = tpe.suggest
# Trials object to track progress
bayes_trials = Trials()

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials)
print("best:", best)
# def f(params):
#     x = params['x']
#     val = x ** 2
#     return {'loss': val, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)
#
# print('best:', best)
#
# print('trials:')
# for trial in trials.trials[:2]:
#     print(trial)

# if __name__ == '__main__':
#     print(objective(params, N_FOLDS))
