import random
import time
from pprint import pprint

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from pyspark.ml.feature import StringIndexer

from pyspark.ml.classification import RandomForestClassifier

from pyspark.sql.types import StructType, StructField, FloatType

import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

MAX_EVALIS = 500
N_FOLDS = 10

conf = SparkConf().setMaster("local[*]").setAppName("mlib")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
schema = StructType([
    StructField("Job_Number", FloatType(), nullable=True),
    StructField("Submit_Time", FloatType(), nullable=True),
    StructField("Wait_Time", FloatType(), nullable=True),
    StructField("Run_Time", FloatType(), nullable=True),
    StructField("NAP", FloatType(), nullable=True),
    StructField("RNP", FloatType(), nullable=True),
    StructField("Status", FloatType(), nullable=True),
    StructField("User_ID", FloatType(), nullable=True),
    StructField("Queue_Number", FloatType(), nullable=True)]
)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# pprint(random_grid)
param_grid = {
    'class_weight': [None, 'balanced'],
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

# plt.hist(param_grid['learning_rate'], color='g', edgecolor='k')
# plt.xlabel('Learning Rate', size=14)
# plt.ylabel('Count', size=14)
# plt.title('Learning Rate Distribution', size=18)
# plt.show()

# 采样结果
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
print(params)
params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
print(params)

rawData = sqlContext.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
data = rawData.toPandas().dropna()
# trainData = data.drop("Job_Number", "NAP", "RNP", "Submit_Time", "User_ID", "Queue_Number")
X = data.loc[:, ["NAP", "Run_Time", "Queue_Number", "Wait_Time", "User_ID"]]  # 特征, "Submit_Time"
Y = data.loc[:, "Status"]  # 目标
# X_1 = MinMaxScaler().fit_transform(X.values.reshape(-1, 1)).reshape(1, -1)[0]

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2022)
train_set = lgb.Dataset(X_train, y_train)
# Cross Validation with Early Stopping in LightGBM 交叉验证
cv_result = lgb.cv(params, train_set, num_boost_round=1000, nfold=10, metrics='auc', early_stopping_rounds=100,
                   verbose_eval=False, seed=50)
# 最高分
cv_results_best = np.max(cv_result['auc-mean'])
# 最高分的标准差
cv_results_std = cv_result['auc-stdv'][np.argmax(['auc-mean'])]

print("the maximum ROC AUC on the validation set was{:.5f} with std of {:.5f}.".format(cv_results_best, cv_results_std))
print('The ideal number of iteractions was {}.'.format(np.argmax(cv_result['auc-mean']) + 1))
print('best n_estimator is {}'.format(len(cv_result['auc-mean'])))
print('best cv score is {}'.format(pd.Series(cv_result['auc-mean']).max()))


# ===========================================================================
# the maximum ROC AUC on the validation set was0.90787 with std of 0.00234.
# The ideal number of iteractions was 1000.
# best n_estimator is 1000
# best cv score is 0.9078703524547699
# ===========================================================================
def random_objective(params, iteration, n_folds=N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    start = time.time()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50)
    end = time.time()
    best_score = np.max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]


random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimator', 'time'],
                              index=list(range(MAX_EVALIS)))
random.seed(50)
# 用前面设置好的值来作为遍历的次数：
for i in range(MAX_EVALIS):
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}  # 随机取样gbm的参数
    print(params)
    if params['boosting_type'] == 'goss':
        params['subsample'] = 1.0
    else:
        params['subsample'] = random.sample(subsample_dist, 1)[0]

    results_list = random_objective(params, i)

    # 将结果添加到数据下一行
    random_results.loc[i, :] = results_list

# 对分析结果进行排序
random_results.sort_values('loss', ascending=True, inplace=True)
random_results.reset_index(inplace=True, drop=True)
print(random_results.head())

# index行标签


# gbm = lgb.LGBMClassifier()
# rf = RandomForestRegressor()
# RandomizedSearchCV
# gbm_random = RandomizedSearchCV(gbm, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42,
# n_jobs=-1)
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
#                                random_state=42,
#                                n_jobs=-1)
# rf_random.fit(X_train, y_train)
# gbm.fit(X_train, y_train,
#                eval_set=[(X_test, y_test)],
#                eval_metric='logloss',
#                callbacks=[lgb.early_stopping(5)])
# gbm_random.fit(X_train, y_train)
# eval_metric默认值：LGBMRegressor 为“l2”，LGBMClassifier 为“logloss”，LGBMRanker 为“ndcg”。
# 使用binary_logloss或者logloss准确率都是一样的。默认logloss

# y_pred = gbm_random.predict(X_test)
# y_pred = gbm.predict(X_test)
# print(gbm_random.best_params_)
# print(rf_random.best_params_)

# accuracy = accuracy_score(y_test, y_pred)
# print("accuarcy: %.2f%%" % (accuracy * 100.0))
