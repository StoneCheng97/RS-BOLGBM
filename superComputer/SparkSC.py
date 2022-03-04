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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD
from six import StringIO
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from tensorflow.python.keras.layers import Dropout

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
# conf = SparkConf().setMaster("local").setAppName("sc")
# sc = SparkContext().getOrCreate(conf)
sq = SQLContext(spark.sparkContext)


def preprocessing(subt=False, rt=False, wt=False):
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
    sjtu_2019_df = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    # sjtu_2019_df = sc.textFile("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv").cache()
    sjtu_2019_pd = sjtu_2019_df.toPandas().dropna()
    # 1.得到标注（目标）
    label = sjtu_2019_pd["Status"]
    # 2.清洗数据
    # 3.特征选择
    X = sjtu_2019_pd.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]  # 特征
    Y = sjtu_2019_pd.loc[:, "Status"]  # 目标
    #
    # 1.过滤思想
    skb = SelectKBest(k=5)
    skb.fit(X, Y)
    # print(skb.transform(X))  # 留下了Submit_Time,Wait_Time和Run_Time，User_ID,Queue_Number去掉了NAP,RNP

    # 特征递归消除(RFE, recursive feature elimination)，SVR(kernel="linear")SVR()就是SVM算法来做回归用的方法（即输入标签是连续值的时候要用的方法）
    lr = LinearRegression()
    ref = RFE(estimator=lr, n_features_to_select=5, step=1)
    # ref = RFE(estimator=SVR(kernel="linear"), n_features_to_select=2, step=1)
    # print(ref.fit_transform(X, Y))  # 留下了NAP,Wait_Time,User_ID,Queue_Number，Run_Time去掉了Submit_Time
    #
    # sjtu_2019_test = sjtu_2019_pd.drop("Status", axis=1) 全特征
    sjtu_2019_test = sjtu_2019_pd.drop(["Status", "NAP", "RNP"], axis=1)
    #
    # 4.特征处理 归一化，标准化
    # scaler_lst = [subt, rt, nap, wt] 全特征
    scaler_lst = [subt, rt, wt]
    # column_lst = ["Submit_Time", "Run_Time", "NAP", "Wait_Time"]
    column_lst = ["Submit_Time", "Run_Time", "Wait_Time"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019_test[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019_test[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019_test[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
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
    models.append(("LGB", LGBMClassifier(boosting_type="goss", n_estimators=400, learning_rate=0.04)))
    # models.append(("KNN", KNeighborsClassifier(n_neighbors=3)))
    # 朴素贝叶斯【生产模型】不适合本实验的数据，对数据要求高
    # models.append(("GaussianNB", GaussianNB()))  # 贝叶斯算法适合离散值
    # models.append(("BernoulliNB", BernoulliNB()))  # 伯努利贝叶斯是二值化
    # 决策树
    # models.append(("DecisionTreeGini", DecisionTreeClassifier()))  # 适合连续值分类
    # models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))  # 适合离散值比较多的分类
    # SVM 效果不好
    # models.append(("SVM Classifier",SVC()))
    # 集成方法
    # 随机森林 目前效果最好   接下来的工作是调参
    # models.append(("RandomForest", RandomForestClassifier(n_estimators=100)))
    # models.append(("RandomForest", RandomForest()))
    # models.append(("RandomForestEntropy", RandomForestClassifier(n_estimators=100, criterion="entropy")))
    # AdaBoost 效果不好，还出UndefinedMetricWarning警告，预测的值中不包含有实际值
    # models.append(("AdaBoost", AdaBoostClassifier(n_estimators=1000)))
    # 逻辑回归 效果不好,数据相关性不强
    # models.append(("LogisticRegression", LogisticRegression(max_iter=10000, C=1000, tol=1e-10,solver="sag")))
    # GBDT 效果还行
    # models.append(("GBDT", GradientBoostingClassifier(max_depth=6, n_estimators=100)))
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]  # 真实label
            Y_pred = clf.predict(X_part)  # 预测label
            print(i)  # 0表示训练集，1表示验证集，2表示测试集

            # 对于类别不均衡的分类模型，采用macro方式会有较大的偏差，采用weighted方式则可较好反映模型的优劣，
            # 因为若类别数量较小则存在蒙对或蒙错的概率，其结果不能真实反映模型优劣，需要较大的样本数量才可计算较为准确的评价值，
            # 通过将样本数量作为权重，可理解为评价值的置信度，数量越多，其评价值越可信
            # https://blog.csdn.net/Urbanears/article/details/105033731
            print(clf_name, "-ACC:", accuracy_score(Y_part, Y_pred))
            print(clf_name, "-Precision:", precision_score(Y_part, Y_pred, average='weighted'))
            print(clf_name, "-REC:", recall_score(Y_part, Y_pred, average='weighted'))
            print(clf_name, "-F-Score:", f1_score(Y_part, Y_pred, average='weighted'))
            # Plotting the results
            fig = plt.figure()
            plt.plot(Y_part[0:2000])
            plt.plot(Y_pred[0:2000])
            plt.title("LightGBM")
            plt.xlabel('Hour')
            plt.ylabel('Electricity load')
            plt.legend(('Actual', 'Predicted'), fontsize='15')
            plt.show()
            # fig.savefig('results/LSTM/final_output.jpg', bbox_inches='tight')

            # # Plot of the loss
            # loss_fig = plt.figure()
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.title('Model Loss')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Validation'], loc='upper left')
            # plt.show()
            # loss_fig.savefig('results/LSTM/final_loss.jpg', bbox_inches='tight')


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


# 聚类
def cluster_test():
    pass


def SparkMlib():
    # data = spark.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
    # datas = data.rdd()
    # RandomForest.trainClassifier(datas,2,{},3,seed=42)
    pass


def main():
    # features, label = preprocessing(subt=True,rt=True, wt=True)  # 标准化处理
    features, label = preprocessing(subt=False, rt=False, wt=False)  # 标准化处理
    # features, label = preprocessing(rt=False, nap=False, wt=False)  # 归一化处理
    # print(preprocessing(rt=True, nap=True, wt=True))
    modeling(features, label)
    # regr_test(features[["NAP", "Wait_Time"]], features["Run_Time"])
    # SparkMlib()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print((end - start) / 60)
