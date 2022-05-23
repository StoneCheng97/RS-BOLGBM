import pandas as pd
import numpy as np
import ReliefF
from numpy import sort
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
# from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 小数点后面保留5位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
# raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu-2019.csv")
test_data = raw_data.dropna()
# features = test_data.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]
# labels = test_data.loc[:, "Status"]
# features, labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values
# 全特征选择
features, labels = test_data.drop(labels=['Status'], axis=1).values, test_data['Status'].values
print("测试集：", test_data)
print('特征：', features)
print("目标：", labels)
print("=================================================================")

ref = ReliefF.ReliefF(n_neighbors=100, n_features_to_keep=5)  # 去掉了Submit_Time
ref.fit(features, labels)
print("ReliefF", ref.transform(features))  # 从高到低排列
print("ReliefF Scores:", ref.feature_scores)
print("ReliefF top_feature:", ref.top_features)
# Relief Scores: [-12957132.   2319008.  -4967888.  19400030.  19289406.  19420496.]
# 按原始顺序排列的，选择的特征是按照分数从高到低排列的
# Relief Scores: [-19124564. -12148496.   2494202.  -5203794.  19331822.  19331822. 19220976.  19354746.]
# Relief top_feature: [7 5 4 6 2 3 1 0]
print("=================================================================")

skb = SelectKBest(score_func=chi2, k=5)  # 去掉了Queue_Number
skb.fit(features, labels)
print("SelectKBest:", skb.transform(features))
print("SelectKBest Scores:", skb.scores_)


# SelectKBest Scores: [6.28210292e+09 5.02352322e+07 1.47366457e+09 2.12300444e+06
#  6.71196507e+04 1.59312948e+03]
# 也是按原数据顺序排列的，但是选择出来的特征不是按分数高低排列的
# SelectKBest Scores: [1.01201681e+08 6.28210292e+09 5.02352322e+07 1.47366457e+09 2.12300444e+06 2.12300444e+06 6.71196507e+04 1.59312948e+03]

# [1.59312948e+03 6.71196507e+04 2.12300444e+06 2.12300444e+06 5.02352322e+07 1.01201681e+08 1.47366457e+09 6.28210292e+09]


# a = [1.01201681e+08, 6.28210292e+09, 5.02352322e+07, 1.47366457e+09, 2.12300444e+06, 2.12300444e+06, 6.71196507e+04,
#      1.59312948e+03]
# print(sort(a))

