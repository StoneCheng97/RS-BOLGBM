import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

import seaborn as sns
import matplotlib.pyplot as plt

# 可视化
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams

sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18, 4)})
rcParams['figure.figsize'] = 18, 4

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# setting random seed for notebook reproducability
rnd_seed = 23
np.random.seed = rnd_seed
np.random.set_state = rnd_seed

spark = SparkSession.builder.master("local[2]").appName("Linear-Regression-California-Housing").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(spark.sparkContext)

HOUSING_DATA = '../datas/cal_housing.data'
# Longitude:指地理位置在地球赤道以北或以南的角距离
# Latitude:指地理位置在地球赤道东侧或西侧的角度距离
# Housing Median Age:属于某一街区人群的年龄中位数。注意，中值是位于观测值频率分布的中点的值
# Total Rooms:是每组房屋中房间的总数
# Total Bedrooms:是每栋房屋中卧室的总数
# Population:是一个街区群的居民数量
# Households:指每栋房屋的单位及其住户
# Median Income:用于登记属于某个区块组的人的收入中位数
# Median House Value:为因变量，指每一组别房屋价值中位数

# 定义模式，对应于CSV数据文件中的一行。
schema = StructType([
    StructField("long", FloatType(), nullable=True),
    StructField("lat", FloatType(), nullable=True),
    StructField("medage", FloatType(), nullable=True),
    StructField("totrooms", FloatType(), nullable=True),
    StructField("totbdrms", FloatType(), nullable=True),
    StructField("pop", FloatType(), nullable=True),
    StructField("houshlds", FloatType(), nullable=True),
    StructField("medinc", FloatType(), nullable=True),
    StructField("medhv", FloatType(), nullable=True)]
)

# Load housing data
housing_df = spark.read.csv(path=HOUSING_DATA, schema=schema).cache()
# 查看前五行
# housing_df.take(5)
# housing_df.show(5)

# run a sample selection
# housing_df.select('pop', 'totbdrms').show(10)

# 根据年龄中位数分组，看分布情况
result_df = housing_df.groupby("medage").count().sort("medage", ascending=False)
# result_df.show(10)
# result_df.toPandas()
# plt.show()

(housing_df.describe().select(
    "summary",
    F.round("medage", 4).alias("medage"),
    F.round("totrooms", 4).alias("totrooms"),
    F.round("totbdrms", 4).alias("totbdrms"),
    F.round("pop", 4).alias("pop"),
    F.round("houshlds", 4).alias("houshlds"),
    F.round("medinc", 4).alias("medinc"),
    F.round("medhv", 4).alias("medhv"))
 .show())
# 查看所有(数值)属性的最小值和最大值。我们看到多个属性具有广泛的值范围:我们将需要规范化您的数据集。
# 有了从小型探索性数据分析中收集到的所有信息，我们就可以对数据进行预处理，将其提供给模型。
#
# 我们不应该关心缺失的值;所有零值都已从数据集中排除。
# 我们可能应该标准化我们的数据，因为我们已经看到最小值和最大值的范围相当大。
# 我们可能还可以添加一些其他属性，比如记录每个房间或每个家庭的房间数量的功能。
# 因变量也很大;为了使我们的生活更轻松，我们必须稍微调整一下数值。

# 首先，让我们从我们的因变量medianHouseValue开始。为方便我们制定目标价值，我们会以10万单位表示房屋价值。这意味着像452600.000000这样的目标应该变成4.526

# 调整`medianHouseValue`
housing_df = housing_df.withColumn("medhv", col("medhv") / 100000)
# housing_df.show(2)

# 特征工程
# 现在我们已经调整了medianHouseValue中的值，现在我们将向数据集添加以下列:
#
# 每户用房数rmsperhh，指每户的用房数;
# 每家每户的人口popperhh，这基本上给了我们一个指标每个街区组的住户中有多少人;
# 每个房间的卧室数bdrmsperrm，这让我们知道每个街区有多少间卧室;

# Add the new columns to `df`
housing_df = (housing_df.withColumn("rmsperhh", F.round(col("totrooms") / col("houshlds"), 2))
              .withColumn("popperhh", F.round(col("pop") / col("houshlds"), 2))
              .withColumn("bdrmsperrm", F.round(col("totbdrms") / col("totrooms"), 2))
              )
# housing_df.show(5)

# Re-order and select columns
housing_df = housing_df.select("medhv",
                               "totbdrms",
                               "pop",
                               "houshlds",
                               "medinc",
                               "rmsperhh",
                               "popperhh",
                               "bdrmsperrm")

# 特征提取，数据归一化
featureCols = ["totbdrms", "pop", "houshlds", "medinc", "rmsperhh", "popperhh", "bdrmsperrm"]
# 将数据放入特征向量列
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
assembled_df = assembler.transform(housing_df)

assembled_df.show(10, truncate=False)

# 标准化 标准化数据=(原数据-均值)/标准差。标准化后的变量值围绕0上下波动，大于0说明高于平均水平，小于0说明低于平均水平。
# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)
scaled_df.select("features", "features_scaled").show(10, truncate=False)

# 建立模型
# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8, .2], seed=rnd_seed)

# 使用ElasticNet弹性网络回归
# 初始化 `lr`
lr = (LinearRegression(featuresCol='features_scaled', labelCol="medhv", predictionCol='predmedhv',
                       maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
linearModel = lr.fit(train_data)

# 评估模型
print(linearModel.coefficients)  # 模型系数

print(linearModel.intercept)  # 模型截距

coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols,
                         "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]

# 预测
# Generate predictions
predictions = linearModel.transform(test_data)
# 提取预测和“已知”正确的标签
predandlabels = predictions.select("predmedhv", "medhv")
predandlabels.show()

# 检查指标

# RMSE(均方根误差)
print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))
# MAE(平均绝对误差)
print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))
# Get the R2
print("R2: {0}".format(linearModel.summary.r2))
# RMSE: 0.8819852045268861
# MAE: 0.6782895319917991
# R2: 0.4180854895364573

# Using the RegressionEvaluator from pyspark.ml package:
# evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='rmse')
# print("RMSE: {0}".format(evaluator.evaluate(predandlabels)))
# evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='mae')
# print("MAE: {0}".format(evaluator.evaluate(predandlabels)))
# evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='r2')
# print("R2: {0}".format(evaluator.evaluate(predandlabels)))

# Using the RegressionMetrics from pyspark.mllib package:
spark.stop()