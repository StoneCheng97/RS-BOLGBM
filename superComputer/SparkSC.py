import time

import lightgbm
import numpy as np
import pandas as pd
# import mmlspark
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers.core import Dense, Activation
# from tensorflow.keras.optimizers import SGD
from six import StringIO
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
# from tensorflow.python.keras.layers import Dropout

from pyspark import SparkContext
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql.context import SQLContext
from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.feature import StandardScaler
# from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.conf import SparkConf
from pyspark.mllib.util import MLUtils
import pyspark
from pyspark.sql import SparkSession

np.set_printoptions(suppress=False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 小数点后面保留5位小数，诸如此类，按需修改吧

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
spark = SparkSession.builder.master("local[*]").appName("SC").getOrCreate()
# spark = SparkSession.builder.master("Spark://master:7077").appName("SC").getOrCreate()
# conf = SparkConf().setMaster("local").setAppName("sc")
# sc = SparkContext().getOrCreate(conf)
sq = SQLContext(spark.sparkContext)


def preprocessing(subt=False, rt=False, wt=False, nap=False, rnp=False):
    schema = StructType([
        StructField("id", FloatType(), nullable=True),
        StructField("Submit_Time", FloatType(), nullable=True),
        StructField("Wait_Time", FloatType(), nullable=True),
        StructField("Run_Time", FloatType(), nullable=True),
        StructField("NAP", FloatType(), nullable=True),
        StructField("RNP", FloatType(), nullable=True),
        StructField("Status", FloatType(), nullable=True),
        StructField("User_ID", FloatType(), nullable=True),
        StructField("Queue_Number", FloatType(), nullable=True)]
    )
    # sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new3.csv", schema=schema).cache()
    sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    # sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\ssc_2018_new.csv", schema=schema).cache()
    # sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\\tc4060_2017_new.csv", schema=schema).cache()
    # sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\ssc_2017_new.csv", schema=schema).cache()
    # sjtu_2019_df = sc.textFile("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv").cache()
    sjtu_2019_pd = sjtu_2019_df.toPandas().dropna()
    print("初始数据集：", sjtu_2019_pd)
    # 1.得到标注（目标）
    label = sjtu_2019_pd["Status"]

    # tc4060_2017_new
    # label = sjtu_2019_pd["Run_Time"]
    # 2.清洗数据
    # 3.特征选择
    # tc4060_2017_new
    # X = sjtu_2019_pd.loc[:, ["Wait_Time", "NAP", "Status", "User_ID", "Queue_Number", "Submit_Time"]]  # 特征
    # Y = sjtu_2019_pd.loc[:, "Run_Time"]  # 目标

    # sjtu_2019_pd
    # X = sjtu_2019_pd.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]  # 特征
    # Y = sjtu_2019_pd.loc[:, "Status"]  # 目标

    # 1.过滤思想
    # skb = SelectKBest(k=5)
    # skb.fit(X, Y)
    # print("SelectKBest:", skb.transform(X))  # 留下了Submit_Time,Wait_Time和Run_Time，User_ID,Queue_Number去掉了NAP,RNP

    # 特征递归消除(RFE, recursive feature elimination)，SVR(kernel="linear")SVR()就是SVM算法来做回归用的方法（即输入标签是连续值的时候要用的方法）
    # lr = LinearRegression()
    # ref = RFE(estimator=lr, n_features_to_select=5, step=1)
    # ref = RFE(estimator=SVR(kernel="linear"), n_features_to_select=2, step=1)
    # print("REF:", ref.fit_transform(X, Y))  # 留下了NAP,Wait_Time,User_ID,Queue_Number，Run_Time去掉了Submit_Time

    # sjtu_2019
    sjtu_2019_test = sjtu_2019_pd.drop("Status", axis=1)  # 全特征
    # print("测试集：",sjtu_2019_test)
    # sjtu_2019_test = sjtu_2019_pd.drop(["id", "Status", "RNP"], axis=1)  # 去掉id和RNP重复列
    print("测试集：", sjtu_2019_test)

    # tc4060_2017_new
    # sjtu_2019_test = sjtu_2019_pd.drop(["Run_Time", "Submit_Time", "RNP"], axis=1)
    #

    # 4.特征处理 归一化，标准化
    # situ_2019
    scaler_lst = [subt, rt, wt, nap, rnp]  # 全特征
    # scaler_lst = [subt, rt, wt, nap]
    column_lst = ["Submit_Time", "Run_Time", "Wait_Time", "NAP", "RNP"]  # 全特征

    # column_lst = ["Submit_Time", "Run_Time", "Wait_Time","NAP"]

    # tc4060_2017_new
    # column_lst = ["Wait_Time", "User_ID", "Wait_Time"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019_test[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019_test[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    print("features：", sjtu_2019_test)
    return sjtu_2019_test, label


def SkbPreprocessing(subt=False, rt=False, wt=False, nap=False):
    schema = StructType([
        StructField("id", FloatType(), nullable=True),
        StructField("Submit_Time", FloatType(), nullable=True),
        StructField("Wait_Time", FloatType(), nullable=True),
        StructField("Run_Time", FloatType(), nullable=True),
        StructField("NAP", FloatType(), nullable=True),
        StructField("RNP", FloatType(), nullable=True),
        StructField("Status", FloatType(), nullable=True),
        StructField("User_ID", FloatType(), nullable=True),
        StructField("Queue_Number", FloatType(), nullable=True)]
    )
    sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    sjtu_2019_pd = sjtu_2019_df.toPandas().dropna()
    print("初始数据集：", sjtu_2019_pd)
    # 1.得到标注（目标）
    label = sjtu_2019_pd["Status"]

    # sjtu_2019
    sjtu_2019_test = sjtu_2019_pd.drop(["Status", "RNP", "User_ID", "Queue_Number"], axis=1)  # SelectKBest选择的五个特征
    print("测试集：", sjtu_2019_test)

    # 4.特征处理 归一化，标准化
    # situ_2019
    # scaler_lst = [subt, rt, wt, nap, rnp]  # 全特征
    scaler_lst = [subt, rt, wt, nap]  # # SelectKBest条件下
    # column_lst = ["Submit_Time", "Run_Time", "Wait_Time", "NAP", "RNP"]  # 全特征
    column_lst = ["Submit_Time", "Run_Time", "Wait_Time", "NAP"]  # # SelectKBest条件下

    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019_test[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019_test[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    print("features：", sjtu_2019_test)
    return sjtu_2019_test, label


def REFPreprocessing(sub=False, rt=False, nap=False, rnp=False, wt=False, uid=False, que=False):
    schema = StructType([
        StructField("id", FloatType(), nullable=True),
        StructField("Submit_Time", FloatType(), nullable=True),
        StructField("Wait_Time", FloatType(), nullable=True),
        StructField("Run_Time", FloatType(), nullable=True),
        StructField("NAP", FloatType(), nullable=True),
        StructField("RNP", FloatType(), nullable=True),
        StructField("Status", FloatType(), nullable=True),
        StructField("User_ID", FloatType(), nullable=True),
        StructField("Queue_Number", FloatType(), nullable=True)]
    )
    sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    sjtu_2019_pd = sjtu_2019_df.toPandas().dropna()
    print("初始数据集：", sjtu_2019_pd)
    # 1.得到标注（目标）
    label = sjtu_2019_pd["Status"]

    # sjtu_2019
    sjtu_2019_test = sjtu_2019_pd.drop(["Status", "id"], axis=1)  # ReliefF选择的五个特征
    print("测试集：", sjtu_2019_test)

    # 4.特征处理 归一化，标准化
    # situ_2019
    # scaler_lst = [subt, rt, wt, nap, rnp]  # 全特征
    scaler_lst = [sub, rt, nap, rnp, wt, uid, que]  # # ReliefF条件下
    # column_lst = ["Submit_Time", "Run_Time", "Wait_Time", "NAP", "RNP"]  # 全特征
    column_lst = ["Submit_Time", "Run_Time", "NAP", "RNP", "Wait_Time", "User_ID", "Queue_Number"]  # # ReliefF

    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019_test[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019_test[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    print("features：", sjtu_2019_test)
    return sjtu_2019_test, label


def RSPreprocessing(subt=False, rt=False, nap=False):
    schema = StructType([
        StructField("id", FloatType(), nullable=True),
        StructField("Submit_Time", FloatType(), nullable=True),
        StructField("Wait_Time", FloatType(), nullable=True),
        StructField("Run_Time", FloatType(), nullable=True),
        StructField("NAP", FloatType(), nullable=True),
        StructField("RNP", FloatType(), nullable=True),
        StructField("Status", FloatType(), nullable=True),
        StructField("User_ID", FloatType(), nullable=True),
        StructField("Queue_Number", FloatType(), nullable=True)]
    )
    sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    sjtu_2019_pd = sjtu_2019_df.toPandas().dropna()
    print("初始数据集：", sjtu_2019_pd)
    # 1.得到标注（目标）
    label = sjtu_2019_pd["Status"]

    # sjtu_2019
    sjtu_2019_test = sjtu_2019_pd.drop(["Status", "RNP", "Wait_Time", "id"], axis=1)  # RS选择的五个特征
    print("测试集：", sjtu_2019_test)

    # 4.特征处理 归一化，标准化
    # situ_2019
    # scaler_lst = [subt, rt, wt, nap, rnp]  # 全特征
    scaler_lst = [subt, rt, nap]  # # RS条件下
    # column_lst = ["Submit_Time", "Run_Time", "Wait_Time", "NAP", "RNP"]  # 全特征
    column_lst = ["Submit_Time", "Run_Time", "NAP"]  # # SelectKBest条件下

    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019_test[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019_test[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    print("features：", sjtu_2019_test)
    return sjtu_2019_test, label


def modeling(features, label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_names = features.columns.values
    l_v = label.values
    # 验证集  X是从特征集中划分验证集和待用集，Y是从label集中划分
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    # 从待用集中划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
    print(len(X_train), len(X_validation), len(X_test))

    # 算法
    # 人工神经网络
    # mdl = Sequential()
    # mdl.add(Dense(50, input_dim=len(f_v[0])))
    # mdl.add(Dropout(0.5))
    # mdl.add(Activation("tanh"))
    #
    # # mdl.add(Dense(500))  # 隐藏层节点500个
    # # mdl.add(Activation('tanh'))
    # # mdl.add(Dropout(0.5))
    # mdl.add(Dense(6))
    # mdl.add(Activation("softmax"))
    # sgd = SGD(lr=0.1)
    # mdl.compile(loss="mean_squared_error", optimizer="adam")  # np.array([[0, 1] if i == 1 else [1, 0] for i in Y_train]
    # mdl.fit(X_train, Y_train, epochs=10000, batch_size=50000)
    # xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    # for i in range(len(xy_lst)):
    #     X_part = xy_lst[i][0]
    #     Y_part = xy_lst[i][1]  # 真实label
    #     Y_pred = np.argmax(mdl.predict(X_part), axis=-1)  # 预测label
    #     print(i)  # 0表示训练集，1表示验证集，2表示测试集
    #     print("NN", "-ACC:", accuracy_score(Y_part, Y_pred))
    #     print("NN", "-Precision:", precision_score(Y_part, Y_pred, average='weighted'))
    #     print("NN", "-REC:", recall_score(Y_part, Y_pred, average='weighted'))
    #     print("NN", "-F-Score:", f1_score(Y_part, Y_pred, average='weighted'))
    # return
    models = []
    # models.append(
    #     ("KNN,SKB调参", KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='kd_tree', leaf_size=40)))
    # models.append(("KNN", KNeighborsClassifier(n_neighbors=9)))
    # 朴素贝叶斯【生产模型】不适合本实验的数据，对数据要求高
    # models.append(("GaussianNB", GaussianNB()))  # 贝叶斯算法适合离散值
    # models.append(("BernoulliNB", BernoulliNB()))  # 伯努利贝叶斯是二值化
    # 决策树
    # models.append(("DecisionTreeGini", DecisionTreeClassifier()))  # 适合连续值分类
    # models.append(("DecisionTreeEntropy,SKB调参",
    #                DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=40,
    #                                       max_features=None)))  # 适合离散值比较多的分类
    # SVM 效果不好
    # models.append(("SVM SGDClassifier", SGDClassifier()))
    # models.append(("SVM LinearSVC", LinearSVC()))
    # 集成方法
    # 随机森林 目前效果最好   接下来的工作是调参
    # models.append(
        # ("RandomForest", RandomForestClassifier()))
        # (
        # "RandomForest,SKB调参", RandomForestClassifier(n_estimators=400, max_depth=30, criterion='gini', bootstrap=True)))
    # ("RandomForest,REF调参", RandomForestClassifier(n_estimators=130, max_depth=32, criterion='gini', bootstrap=False)))
    # ("RandomForest,REF调参",
    #  RandomForestClassifier(n_estimators=211, max_depth=20, criterion='gini', bootstrap=False)))
    # models.append(("RandomForest", RandomForest()))
    # models.append(("RandomForestEntropy",
    # RandomForestClassifier(n_estimators=171, criterion="entropy", max_depth=None, max_features=5)))
    # AdaBoost 效果不好，还出UndefinedMetricWarning警告，预测的值中不包含有实际值
    # models.append(("AdaBoost", AdaBoostClassifier(n_estimators=1000)))
    # 逻辑回归 效果不好,数据相关性不强
    # models.append(("LogisticRegression", LogisticRegression(max_iter=10000, C=1000, tol=1e-10,solver="sag")))
    # GBDT 效果还行
    # models.append(("GBDT,SKB调参", GradientBoostingClassifier()))
    # models.append(("GBDT,SKB调参", GradientBoostingClassifier(max_depth=6, n_estimators=312, learning_rate=0.08452088679707226,
    #                                               criterion='friedman_mse')))  # learning_rate=0.12605847206065268
    # models.append(("GBDT,REF调参", GradientBoostingClassifier(max_depth=10, n_estimators=283, learning_rate=0.13824951937907706,
    #                                               criterion='friedman_mse')))  # learning_rate=0.12605847206065268
    # models.append(("GBDT", GradientBoostingClassifier(max_depth=10, n_estimators=279, learning_rate=0.015477153246310426,
    #                                                   criterion='friedman_mse')))  # learning_rate=0.12605847206065268
    # models.append(("LGBM", LGBMClassifier(boosting_type="goss", n_estimators=400, learning_rate=0.04)))
    # models.append(
    #     ("LGBM,SKB调参", LGBMClassifier(boosting_type="dart", n_estimators=400, learning_rate=0.19706765150537875,
    #                                   num_leaves=31, min_child_samples=330, subsample_for_bin=20000)))
    models.append(
        ("LGBM,REF调参", LGBMClassifier(boosting_type="goss", n_estimators=383, learning_rate=0.048021192742094972,
                                      num_leaves=35, min_child_samples=200, subsample_for_bin=140000)))
    # models.append(("LGBM调参", LGBMClassifier(boosting_type="goss", n_estimators=227, learning_rate=0.17575260374540685,
    #                                         num_leaves=146, min_child_samples=225, subsample_for_bin=80000)))
    # models.append(("LGBM无调参", LGBMClassifier(boosting_type="goss")))
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            # print(X_part)
            Y_part = xy_lst[i][1]  # 真实label
            Y_pred = clf.predict(X_part)  # 预测label
            # Y_pred2 = clf.predict_proba(X_part)  # 预测label
            print(i)  # 0表示训练集，1表示验证集，2表示测试集

            # 对于类别不均衡的分类模型，采用macro方式会有较大的偏差，采用weighted方式则可较好反映模型的优劣，
            # 因为若类别数量较小则存在蒙对或蒙错的概率，其结果不能真实反映模型优劣，需要较大的样本数量才可计算较为准确的评价值，
            # 通过将样本数量作为权重，可理解为评价值的置信度，数量越多，其评价值越可信
            # https://blog.csdn.net/Urbanears/article/details/105033731
            print(clf_name, "-ACC:", accuracy_score(Y_part, Y_pred))
            # print(clf_name, "-AUC:", roc_auc_score(Y_part, Y_pred, multi_class='ovr'))
            print(clf_name, "-Precision:", precision_score(Y_part, Y_pred, average='weighted'))
            print(clf_name, "-REC:", recall_score(Y_part, Y_pred, average='weighted'))
            print(clf_name, "-F-Score:", f1_score(Y_part, Y_pred, average='weighted'))
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\ReliefF\\true_values_" + str(i) + clf_name,
            #            Y_part)
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\ReliefF\predict_values_" + str(i) + clf_name,
            #            Y_pred)
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\SelectKBest\\true_values_" + str(i) + clf_name,
            #            Y_part)
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\SelectKBest\predict_values_" + str(i) + clf_name,
            #            Y_pred)
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\\full\\true_values_" + str(i) + clf_name,
            #            Y_part)
            # np.savetxt("D:\pythonProject\spark\superComputer\\results\\full\predict_values_" + str(i) + clf_name,
            #            Y_pred)


# 回归
def regr_test(features, label):
    print("X", features)
    print("Y", label)
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    regr = LinearRegression()  # Coef参数 [0.01521006 0.07334342] MSE: 0.99439585
    # 岭回归
    # regr = Ridge(alpha=0.6)
    # Lasso
    # regr = Lasso(alpha=0.01)
    regr.fit(features.values, label.values)
    Y_pred = regr.predict(features.values)
    print("Coef参数", regr.coef_)
    from sklearn.metrics import mean_squared_error
    print("MSE:", mean_squared_error(Y_pred, label.values))


def pick_arange(arange, num):
    if num > len(arange):
        print('num out of length')
    else:
        output = np.array([], dtype=arange.dtype)
        seg = len(arange) / num
        for n in range(num):
            if int(seg * (n + 1)) >= len(arange):
                output = np.append(output, arange[-1])
            else:
                output = np.append(output, arange[int(seg * n)])
        return output


# 聚类
def cluster_test():
    Y_part = pd.read_csv('D:\pythonProject\spark\superComputer\\true_values')
    Y_pred = pd.read_csv('D:\pythonProject\spark\superComputer\predict_values')
    print(Y_part)
    print("-AUC:", roc_auc_score(Y_part, Y_pred, multi_class='ovo'))
    print("-ACC:", accuracy_score(Y_part, Y_pred))


def SparkMlib():
    # data = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    # datas = data.rdd()
    # RandomForest.trainClassifier(datas,2,{},3,seed=42)
    pass


def main():
    # features, label = preprocessing(subt=False, rt=False, wt=False, nap=False, rnp=False)  # 全特征minmax归一化处理
    # features, label = preprocessing(subt=True, rt=True, wt=True, nap=True, rnp=True)  # 全特征z-score归一化标准化处理
    features, label = RSPreprocessing(subt=True, rt=True, nap=True)  # 全特征z-score归一化标准化处理
    # features, label = SkbPreprocessing(subt=True, rt=True, wt=True, nap=True)  # SKBz-score归一化标准化处理
    # features, label = SkbPreprocessing(subt=False, rt=False, wt=False, nap=False)  # SKB minmax归一化标准化处理
    # features, label = REFPreprocessing(sub=False, rt=False, nap=False, rnp=False, wt=False, uid=False,
    #                                    que=False)  # ReliefF minmax归一化标准化处理 效果好
    # features, label = REFPreprocessing(sub=True, rt=True, nap=True, rnp=True, wt=True, uid=True,
    #                                    que=True)  # ReliefF z-score归一化标准化处理
    modeling(features, label)
    # cluster_test()


if __name__ == '__main__':
    start = time.time()
    main()
    # a = np.arange(0, 10)
    # print(pick_arange(a, 5))
    end = time.time()
    print("时间",(end - start) / 60)
