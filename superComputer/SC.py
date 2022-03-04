import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from mpmath import norm
import pydotplus
from IPython import display

display.set_matplotlib_formats('svg')

os.environ["PATH"] += os.pathsep + "G:/Graphviz/bin/"
np.set_printoptions(suppress=False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 小数点后面保留3位小数，诸如此类，按需修改吧

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

path1 = "D:\pythonProject\spark\datas\chipotle.tsv"
path2 = "D:\pythonProject\spark\datas\Euro2012_stats.csv"
path3 = "D:\pythonProject\spark\datas\wxsc.txt"
path4 = "D:\pythonProject\spark\datas\sc\sjtu-2019.csv"
path5 = "D:\pythonProject\spark\datas\sc\ssc_2019.csv"
path6 = "D:\pythonProject\spark\datas\sc\ssc_2018.csv"
path7 = "D:\pythonProject\spark\datas\sc\ssc_20170412.csv"
path8 = "D:\pythonProject\spark\datas\sc//tc4060_2019.csv"
path9 = "D:\pythonProject\spark\datas\sc//tc4060.csv"
path10 = "D:\pythonProject\spark\datas\sc//tc4060_2018.csv"
path11 = "D:\pythonProject\spark\datas\sc\wxsc_2017.csv"


# False 归一化，True 标准化
def preprocessing(subt=False, rt=False, nap=False, wt=False):
    sjtu_2019 = pd.read_csv(path4).drop(
        ['Job_Number', 'RM', 'RT', 'ACTU', 'Used_Memory', 'Executable (Application) Number', 'Group_ID',
         'Partition_Number',
         'Preceding Job Number', 'Think Time from Preceding Job'], axis=1)
    ssc_2019 = pd.read_csv(path5)
    ssc_2018 = pd.read_csv(path6)
    ssc_2017 = pd.read_csv(path7)
    tc4060_2019 = pd.read_csv(path8)
    tc4060_2017 = pd.read_csv(path9)
    tc4060_2018 = pd.read_csv(path10)
    wxsc_2017 = pd.read_csv(path11)
    # 1 年=31536000 秒，过滤时间为负数和运行时间大于1.5年的数据
    ssc_2018_1 = ssc_2018[ssc_2018['Run_Time'] > 0]
    tc4060_2019_1 = tc4060_2019[tc4060_2019['Run_Time'] > 0]
    tc4060_2017_1 = tc4060_2017[tc4060_2017['Wait_Time'] > 0]
    tc4060_2017_2 = tc4060_2017_1[tc4060_2017_1['Run_Time'] < 47304000]

    # 过滤取对数后仍为负数的行
    sjtu_2019_1 = sjtu_2019[np.log(sjtu_2019['Run_Time']) > 0]
    ssc_2019_1 = ssc_2019[np.log(ssc_2019['Run_Time']) > 0]
    ssc_2017_1 = ssc_2017[np.log(ssc_2017['Run_Time']) > 0]
    tc4060_2017_3 = tc4060_2017_2[np.log(tc4060_2017_2['Run_Time']) > 0]
    wxsc_2017_1 = wxsc_2017[np.log(wxsc_2017['Run_Time']) > 0]

    # 将Status状态值大于5的都转换成运行成功的状态值1【待定】

    sjtu_2019.loc[sjtu_2019['Status'] > 5, 'Status'] = 1
    ssc_2017.loc[ssc_2017['Status'] > 5, 'Status'] = 1

    # 1.得到标注（目标）
    label = sjtu_2019["Status"]
    # 2.清洗数据
    # 3.特征选择
    # Submit_Time,Wait_Time和Run_Time，User_ID,Queue_Number五个特征最高95点几
    # Wait_Time和Run_Time，NAP,User_ID,Queue_Number五个特征最高87点几
    sjtu_2019 = sjtu_2019.drop(["Status", "NAP", "RNP"], axis=1)
    # sjtu_2019 = sjtu_2019.drop(["Status", "Submit_Time", "RNP"], axis=1)

    # 4.特征处理 归一化，标准化
    # scaler_lst = [rt, nap, wt]
    scaler_lst = [subt, rt, wt]
    column_lst = ["Submit_Time","Run_Time","Wait_Time"]
    # column_lst = ["Run_Time", "NAP", "Wait_Time"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            sjtu_2019[column_lst[i]] = \
                MinMaxScaler().fit_transform(sjtu_2019[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            sjtu_2019[column_lst[i]] = \
                StandardScaler().fit_transform(sjtu_2019[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    return sjtu_2019, label


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
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.svm import SVC
    from six import StringIO
    from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
    models = []
    models.append(("KNN", KNeighborsClassifier(n_neighbors=3)))
    # 朴素贝叶斯【生产模型】不适合本实验的数据，对数据要求高
    models.append(("GaussianNB", GaussianNB()))  # 贝叶斯算法适合离散值
    models.append(("BernoulliNB", BernoulliNB()))  # 伯努利贝叶斯是二值化
    # 决策树
    models.append(("DecisionTreeGini", DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))
    # SVM 效果不好
    # models.append(("SVM Classifier", SVC()))
    # 随机森林 目前效果最好   接下来的工作是调参
    models.append(("RandomForest", RandomForestClassifier()))
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)  # 0表示训练集，1表示验证集，2表示测试集
            print(clf_name, "-ACC:", accuracy_score(Y_part, Y_pred))
            print(clf_name, "-REC:", recall_score(Y_part, Y_pred, average='weighted'))
            print(clf_name, "-F-Score:", f1_score(Y_part, Y_pred, average='weighted'))
            # 画决策树
            # dot_data = export_graphviz(clf, out_file=None, feature_names=f_names,
            #                            filled=True, rounded=True)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf("dt_tree.pdf")

    # 标准化 测试集结果略低于训练集，说明有过拟合现象出现，归一化，结果和标准化差不多
    #     保存模型
    # joblib.dump(knn_clf, "knn_clf_model")


#     使用模型
#     knn_clf = joblib.load("knn_clf")


def main():
    features, label = preprocessing(subt=True, rt=True, wt=True)  # 标准化处理
    # features, label = preprocessing(rt=True, nap=True, wt=True)
    # features, label = preprocessing(rt=False, nap=False, wt=False)  # 归一化处理
    print(preprocessing(subt=True,rt=True,wt=True))
    # print(preprocessing(rt=True, nap=True, wt=True))
    modeling(features, label)


if __name__ == '__main__':
    main()
