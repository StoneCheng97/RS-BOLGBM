from IPython.core.display import display

import pyspark
import synapse

from pyspark import SparkConf, SparkContext
import os

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load("D:\pythonProject\spark\datas\sc\data.csv")
# print("records read: " + str(df.count()))
# print("Schema: ")
# df.printSchema()
train, test = df.randomSplit([0.85, 0.15], seed=1)
from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[1:]
featurizer = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)
train_data = featurizer.transform(train)['Bankrupt?', 'features']
test_data = featurizer.transform(test)['Bankrupt?', 'features']
display(train_data.groupBy("Bankrupt?").count())

from synapse.ml.lightgbm import LightGBMClassifier

model = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True)
model = model.fit(train_data)
