import pandas as pd
import numpy as np
import ReliefF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
# from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.set_printoptions(suppress=False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 小数点后面保留5位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
test_data = raw_data.dropna()
# features = test_data.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]
# labels = test_data.loc[:, "Status"]
features, labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values
print("测试集：", test_data)
print('特征：', features)
print("目标：", labels)
print("=================================================================")

ref = ReliefF.ReliefF(n_neighbors=100, n_features_to_keep=5)  # 去掉了Submit_Time
ref.fit(features, labels)
print("Relief", ref.transform(features))
print("Relief Scores:", ref.feature_scores)
print("Relief top_feature:", ref.top_features)
# Relief Scores: [-12957132.   2319008.  -4967888.  19400030.  19289406.  19420496.]
print("=================================================================")

skb = SelectKBest(score_func=chi2, k=5)  # 去掉了Queue_Number
skb.fit(features, labels)
print("SelectKBest:", skb.transform(features))
print("SelectKBest Scores:", skb.scores_)
# SelectKBest Scores: [6.28210292e+09 5.02352322e+07 1.47366457e+09 2.12300444e+06
#  6.71196507e+04 1.59312948e+03]
