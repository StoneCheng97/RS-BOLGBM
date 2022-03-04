# from IPython.core.display import display
#
# from synapse.ml.lightgbm import LightGBMClassifier
# import os
#
# from pyspark.sql import SparkSession
# from pyspark.conf import SparkConf
# from pyspark.ml.feature import VectorAssembler
#
# # conf = SparkConf().setMaster("local[*]").setAppName("sc")
# # spark = SparkSession.builder.master("local[*]").appName("MMLSpark") \
# #     .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5") \
# #     .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
# #     .getOrCreate()
#
# spark = SparkSession.builder.master("local").appName("sc").config(conf=SparkConf()).getOrCreate()
# df = spark.read.format("csv") \
#     .option("header", True) \
#     .option("inferSchema", True) \
#     .load("D:\pythonProject\spark\datas\sc\data.csv")
# # print("records read: " + str(df.count()))
# # print("Schema: ")
# # df.printSchema()
# train, test = df.randomSplit([0.85, 0.15], seed=1)
#
# feature_cols = df.columns[1:]
# featurizer = VectorAssembler(
#     inputCols=feature_cols,
#     outputCol='features'
# )
# train_data = featurizer.transform(train)['Bankrupt?', 'features']
# test_data = featurizer.transform(test)['Bankrupt?', 'features']
# display(train_data.groupBy("Bankrupt?").count())
#
# model = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True)
# model = model.fit(train_data)
import pyspark

spark = pyspark.sql.SparkSession.builder.master("local[*]") \
    .appName("mmlspark") \
    .config("spark.jars.packages",
            "com.microsoft.ml.spark:mmlspark_2.12:1.0.0-rc3-76-aad223e0-SNAPSHOT,org.apache.hadoop:hadoop-azure:3.3.1") \
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
    .config("spark.executor.memory", "4g") \
    .enableHiveSupport() \
    .getOrCreate()
df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/company_bankruptcy_prediction_data.csv")
# print dataset size
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
import mmlspark
