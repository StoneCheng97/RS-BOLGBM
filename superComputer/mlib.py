import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pyspark.ml.feature import StringIndexer

from pyspark.ml.classification import RandomForestClassifier

from pyspark.sql.types import StructType, StructField, FloatType

import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

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
rawData = sqlContext.read.csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv", schema=schema).cache()
data = rawData.toPandas().dropna()
# trainData = data.drop("Job_Number", "NAP", "RNP", "Submit_Time", "User_ID", "Queue_Number")
X = data.loc[:, ["NAP", "Run_Time", "Queue_Number", "Wait_Time", "User_ID"]]  # 特征, "Submit_Time"
Y = data.loc[:, "Status"]  # 目标
# X_1 = MinMaxScaler().fit_transform(X.values.reshape(-1, 1)).reshape(1, -1)[0]

print(X)
print(Y)
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2022)

gbm = lgb.LGBMClassifier()
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(5)])
# eval_metric默认值：LGBMRegressor 为“l2”，LGBMClassifier 为“logloss”，LGBMRanker 为“ndcg”。
# 使用binary_logloss或者logloss准确率都是一样的。默认logloss

y_pred = gbm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuarcy: %.2f%%" % (accuracy * 100.0))
