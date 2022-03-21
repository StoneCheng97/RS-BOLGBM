import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
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
features, labels = test_data.drop(labels=['Status','RNP'], axis=1).values, test_data['Status'].values
print('特征：', features)
print("目标：", labels)

# skb = SelectKBest(score_func=chi2, k=5)# 去掉了Queue_Number
# skb.fit(features, labels)
# print("SelectKBest:", skb.transform(features))
# print("Scores:",skb.scores_)
ref = ReliefF(n_features_to_select=2, n_neighbors=100)
ref.fit(features, labels)
print(ref.transform(features))

# clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100), RandomForestClassifier(n_estimators=100))
# print(np.mean(cross_val_score(clf, features, labels)))
