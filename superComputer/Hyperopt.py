import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# 数据
from sklearn.preprocessing import normalize, scale
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv").loc[120000:125000]
test_data = data.dropna()
# X = test_data.drop(labels='Status', axis=1).values # 全特征
# X = test_data.drop(labels=['Status', 'RNP', 'User_ID', 'Queue_Number'], axis=1).values  # SelectKBest选择的特征
X = test_data.drop(labels=['Status', 'RNP', 'User_ID', 'Queue_Number'], axis=1).values  # ReliefF选择的特征
y = test_data['Status'].values
print(test_data)
print(X)
print(y)


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

def hyperopt_train_test_KNN(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()


def hyperopt_train_test_DT(params):
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y).mean()


def hyperopt_train_test_RF(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()


def hyperopt_train_test_SVM(params):
    clf = LinearSVC(**params)
    return cross_val_score(clf, X, y).mean()


def hyperopt_train_test_GDBT(params):
    clf = GradientBoostingClassifier(**params)
    return cross_val_score(clf, X, y).mean()


space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
    # best:
    # {'n_neighbors': 97}
}
space4DT = {
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5)),
    'criterion': hp.choice('criterion', {"gini", "entropy"})
    #     {'criterion': 0, 'max_depth': 0, 'max_features': 2}
}
space4RF = {
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5)),
    'n_estimators': hp.choice('n_estimators', range(100, 500)),
    'criterion': hp.choice('criterion', {"gini", "entropy"}),
}
space4SVM = {
    'penalty': hp.choice('penalty', ["l1", "l2"]),
    'C': hp.uniform('C', 0, 20),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}
space4GDBT = {
    'learning_rate': hp.uniform('learning_rate', 0, 0.2),
    'n_estimators': hp.choice('n_estimators', range(100, 500)),
    'criterion': hp.choice('criterion', ["friedman_mse", "squared_error"]),

}


# best = 0


def f(params):
    # global best
    # acc = hyperopt_train_test_KNN(params)
    # acc = hyperopt_train_test_DT(params)
    acc = hyperopt_train_test_RF(params)
    # acc = hyperopt_train_test_SVM(params)
    # acc = hyperopt_train_test_GDBT(params)
    # if acc > best:
    #     best = acc
    # print('new best:', best, params)
    #
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# best = fmin(f, space4DT, algo=tpe.suggest, max_evals=100, trials=trials)
best = fmin(f, space4RF, algo=tpe.suggest, max_evals=100, trials=trials)
# best = fmin(f, space4SVM, algo=tpe.suggest, max_evals=100, trials=trials)
# best = fmin(f, space4GDBT, algo=tpe.suggest, max_evals=100, trials=trials)
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:')
print(best)
