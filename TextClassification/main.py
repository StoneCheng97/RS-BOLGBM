import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

# Spark环境
hc = (SparkSession.builder
      .appName('Toxic Comment Classification')
      .enableHiveSupport()
      .config("spark.executor.memory", "4G")
      .config("spark.driver.memory", "18G")
      .config("spark.executor.cores", "7")
      .config("spark.python.worker.memory", "4G")
      .config("spark.driver.maxResultSize", "0")
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.default.parallelism", "2")
      .getOrCreate())
hc.sparkContext.setLogLevel('ERROR')


# print(hc.version)
def to_spark_df(fin):
    """
    Parse a filepath to a spark dataframe using the pandas api.

    Parameters
    ----------
    fin : str
        The path to the file on the local filesystem that contains the csv data.

    Returns
    -------
    df : pyspark.sql.dataframe.DataFrame
        A spark DataFrame containing the parsed csv data.
    """
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = hc.createDataFrame(df)
    return (df)


# Load the train-test sets
train = to_spark_df("../datas/train.csv")
test = to_spark_df("../datas/test.csv")

out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]

# Sadly the output is not as  pretty as the pandas.head() function
# train.show(5)

# View some toxic comments, 查看一些恶评
# train.filter(F.col('toxic') == 1).show(5)

# 基本句子分词器
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
wordsData = tokenizer.transform(train)

# 计算文档中的字数[词频]
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)

tf.select('rawFeatures').take(2)

# 建立idf模型，将原始的token频率转换为tf-idf对应的token频率
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

tfidf.select("features").first()

# 首先使用LogisticRegression类进行测试。我喜欢先创建对象的实例，以检查它们的方法和文档字符串，并弄清楚如何访问数据。
# 建立二元toxic列的logistic回归模型。使用features列(tfidf值)作为输入向量X，使用toxic列作为输出向量y。

REG = 0.1
lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)
tfidf.show(5)

lrModel = lr.fit(tfidf.limit(5000))
res_train = lrModel.transform(tfidf)
res_train.select("id", "toxic", "probability", "prediction").show(20)

res_train.show(5)

# 创建一个用户定义函数(udf)来选择列向量每一行中的第二个元素
extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
(res_train.withColumn("proba", extract_prob("probability"))
 .select("proba", "prediction")
 .show())

# Create the results DataFrame
# 转换test数据文本
test_tokens = tokenizer.transform(test)
test_tf = hashingTF.transform(test_tokens)
test_tfidf = idfModel.transform(test_tf)

# # 用id列初始化新的DataFrame
# test_res = test.select('id')
# test_res.head()
#
# # 对每个类别进行预测
# test_probs = []
# for col in out_cols:
#     print(col)
#     lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
#     print("...fitting")
#     lrModel = lr.fit(tfidf)
#     print("...predicting")
#     res = lrModel.transform(test_tfidf)
#     print("...appending result")
#     test_res = test_res.join(res.select('id', 'probability'), on="id")
#     print("...extracting probability")
#     test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
#     test_res.show(5)

# toxic
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+
# |              id|       toxic|
# +----------------+------------+
# |000968ce11f5ee34|  0.04655437|
# |00491682330fdd1d|3.6486778E-8|
# |008eb47c4684d190|   0.6308229|
# |00d251f47486b6d2|  0.06102414|
# |0114ae82c53101a9|  0.43038085|
# +----------------+------------+
# only showing top 5 rows
#
# severe_toxic
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+------------+
# |              id|       toxic|severe_toxic|
# +----------------+------------+------------+
# |000968ce11f5ee34|  0.04655437| 0.007900135|
# |00491682330fdd1d|3.6486778E-8| 9.672807E-7|
# |008eb47c4684d190|   0.6308229|  0.00130351|
# |00d251f47486b6d2|  0.06102414| 0.007802302|
# |0114ae82c53101a9|  0.43038085|  0.07279135|
# +----------------+------------+------------+
# only showing top 5 rows
#
# obscene
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+------------+------------+
# |              id|       toxic|severe_toxic|     obscene|
# +----------------+------------+------------+------------+
# |000968ce11f5ee34|  0.04655437| 0.007900135|  0.03559102|
# |00491682330fdd1d|3.6486778E-8| 9.672807E-7| 2.616639E-8|
# |008eb47c4684d190|   0.6308229|  0.00130351|0.0024817144|
# |00d251f47486b6d2|  0.06102414| 0.007802302|  0.03433142|
# |0114ae82c53101a9|  0.43038085|  0.07279135|  0.29547474|
# +----------------+------------+------------+------------+
# only showing top 5 rows
#
# threat
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+------------+------------+------------+
# |              id|       toxic|severe_toxic|     obscene|      threat|
# +----------------+------------+------------+------------+------------+
# |000968ce11f5ee34|  0.04655437| 0.007900135|  0.03559102| 0.002523175|
# |00491682330fdd1d|3.6486778E-8| 9.672807E-7| 2.616639E-8|2.2700187E-4|
# |008eb47c4684d190|   0.6308229|  0.00130351|0.0024817144| 0.001067896|
# |00d251f47486b6d2|  0.06102414| 0.007802302|  0.03433142|0.0023257558|
# |0114ae82c53101a9|  0.43038085|  0.07279135|  0.29547474|0.0042685755|
# +----------------+------------+------------+------------+------------+
# only showing top 5 rows
#
# insult
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+------------+------------+------------+------------+
# |              id|       toxic|severe_toxic|     obscene|      threat|      insult|
# +----------------+------------+------------+------------+------------+------------+
# |000968ce11f5ee34|  0.04655437| 0.007900135|  0.03559102| 0.002523175| 0.029229807|
# |00491682330fdd1d|3.6486778E-8| 9.672807E-7| 2.616639E-8|2.2700187E-4|1.9096854E-6|
# |008eb47c4684d190|   0.6308229|  0.00130351|0.0024817144| 0.001067896|0.0153904725|
# |00d251f47486b6d2|  0.06102414| 0.007802302|  0.03433142|0.0023257558| 0.036742307|
# |0114ae82c53101a9|  0.43038085|  0.07279135|  0.29547474|0.0042685755|  0.19512838|
# +----------------+------------+------------+------------+------------+------------+
# only showing top 5 rows
#
# identity_hate
# ...fitting
# ...predicting
# ...appending result
# ...extracting probability
# +----------------+------------+------------+------------+------------+------------+-------------+
# |              id|       toxic|severe_toxic|     obscene|      threat|      insult|identity_hate|
# +----------------+------------+------------+------------+------------+------------+-------------+
# |000968ce11f5ee34|  0.04655437| 0.007900135|  0.03559102| 0.002523175| 0.029229807|  0.006218153|
# |00491682330fdd1d|3.6486778E-8| 9.672807E-7| 2.616639E-8|2.2700187E-4|1.9096854E-6|  8.143212E-6|
# |008eb47c4684d190|   0.6308229|  0.00130351|0.0024817144| 0.001067896|0.0153904725| 0.0020708058|
# |00d251f47486b6d2|  0.06102414| 0.007802302|  0.03433142|0.0023257558| 0.036742307|  0.006709278|
# |0114ae82c53101a9|  0.43038085|  0.07279135|  0.29547474|0.0042685755|  0.19512838|  0.013183156|
# +----------------+------------+------------+------------+------------+------------+-------------+
# only showing top 5 rows