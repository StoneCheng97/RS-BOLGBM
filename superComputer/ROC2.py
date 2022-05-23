import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
# raw_data = pd.read_csv("D:\pythonProject\spark\datas\sc\sjtu-2019.csv")
test_data = raw_data.dropna()
# features = test_data.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]
# labels = test_data.loc[:, "Status"]
# features, labels = test_data.drop(labels=['id', 'Status', 'RNP'], axis=1).values, test_data['Status'].values
# 全特征选择
features, labels = test_data.drop(labels=['Status'], axis=1).values, test_data['Status'].values
# X, y = load_wine(return_X_y=True)
# y = y == 2

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42)
# svc = SVC(random_state=42)
# svc.fit(X_train, y_train)
#
# svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
# plt.show()
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
# svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
