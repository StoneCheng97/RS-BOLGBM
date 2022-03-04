import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

np.set_printoptions(suppress=False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # 小数点后面保留3位小数，诸如此类，按需修改吧
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

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
# 1 年=31536000 秒，去掉时间为负数和运行时间大于1.5年的数据
ssc_2018_1 = ssc_2018[ssc_2018['Run_Time'] > 0]
tc4060_2019_1 = tc4060_2019[tc4060_2019['Run_Time'] > 0]
tc4060_2017_1 = tc4060_2017[tc4060_2017['Wait_Time'] > 0]
tc4060_2017_2 = tc4060_2017_1[tc4060_2017_1['Run_Time'] < 47304000]

# 去掉取对数后仍为负数的行
sjtu_2019_1 = sjtu_2019[np.log(sjtu_2019['Run_Time']) > 0]
ssc_2019_1 = ssc_2019[np.log(ssc_2019['Run_Time']) > 0]
ssc_2017_1 = ssc_2017[np.log(ssc_2017['Run_Time']) > 0]
tc4060_2017_3 = tc4060_2017_2[np.log(tc4060_2017_2['Run_Time']) > 0]
wxsc_2017_1 = wxsc_2017[np.log(wxsc_2017['Run_Time']) > 0]

# 将Status状态值大于5的都转换成运行成功的状态值1【此字段待定】

sjtu_2019.loc[sjtu_2019['Status'] > 5, 'Status'] = 1
sjtu_2019.to_csv("D:\pythonProject\spark\datas\sc\sjtu_2019_new.csv")
# ssc_2019.loc[ssc_2019['Status'] > 5, 'Status'] = 1
# ssc_2018_1.loc[ssc_2018_1['Status'] > 5, 'Status'] = 1
ssc_2017.loc[ssc_2017['Status'] > 5, 'Status'] = 1

fig, ax = plt.subplots(figsize=(10, 10))
scatter_matrix(tc4060_2019_1[['Run_Time', 'NAP', 'Status']], alpha=0.2, diagonal='hist', ax=ax)
plt.show()

# X = sjtu_2019.loc[:, ["Wait_Time", "NAP", "Run_Time", "User_ID", "Queue_Number", "Submit_Time"]]  # 特征
# Y = sjtu_2019.loc[:, "Status"]  # 目标
# #
# # 1.过滤思想
# skb = SelectKBest(k=5)
# skb.fit(X, Y)
# print(skb.transform(X))  # 留下了Submit_Time,Wait_Time和Run_Time，User_ID,Queue_Number去掉了NAP,RNP
#
# # 特征递归消除(RFE, recursive feature elimination)，SVR(kernel="linear")SVR()就是SVM算法来做回归用的方法（即输入标签是连续值的时候要用的方法）
# lr = LinearRegression()
# ref = RFE(estimator=lr, n_features_to_select=5, step=1)
# ref = RFE(estimator=SVR(kernel="linear"), n_features_to_select=2, step=1)
# print(ref.fit_transform(X, Y))  # 留下了NAP,Wait_Time,User_ID,Queue_Number，Run_Time去掉了Submit_Time

# 2.嵌入思想
# sfm = SelectFromModel(estimator=DecisionTreeRegressor(), threshold="阈值")  # 阈值的选择很重要
# print(sfm.fit_transform(X, Y))

# # 用户4的运行情况
# data = sjtu_2019[(sjtu_2019.User_ID==4)]
# print(data.describe())


# 异常值处理
# 异常值分析
# （1）3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值 → p(|x - μ| > 3σ) ≤ 0.003
# data = sjtu_2019['Run_Time'] / 360
# u = data.mean()
# std = data.std()
# ss.kstest(data, 'norm', (u, std))
# print('均值为：%.3f，标准差为：%.3f' % (u, std))
# print('------')
# # 正态性检验
# fig = plt.figure(figsize=(10, 6))
# ax1 = fig.add_subplot(2, 1, 1)
# data.plot(kind='kde', grid=True, style='-k', title='密度曲线')
#
# # 绘制数据密度曲线
# ax2 = fig.add_subplot(2, 1, 2)
# error = data[np.abs(data - u) > 3 * std]
# data_c = data[np.abs(data - u) <= 3 * std]
# print('异常值共%i条' % len(error))
#
# # 筛选出异常值error、剔除异常值之后的数据data_c
#
# plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
# plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)
# # plt.xlim([-10,10010])
# plt.grid()
# plt.show()
# 图表表达
# data = pd.DataFrame({'sjtu': sjtu_2019["Run_Time"], 'ssc_2019': ssc_2019["Run_Time"]})
#
# data['sjtu_per']=data['sjtu']/data['sjtu'].sum()
# data['ssc_per']=data['ssc_2019']/data['ssc_2019'].sum()
#
# data['sjtu_per%'] = data['sjtu_per'].apply(lambda x: '%.5f%%' % (x*100))
# data['ssc_per%'] = data['ssc_per'].apply(lambda x: '%.5f%%' % (x*100))
# print(data.head(20))
#
# fig,axes = plt.subplots(2,1,figsize = (10,6),sharex=True)
# data[['sjtu','ssc_2019']].plot(kind='line',style = '--.',alpha = 0.8,ax=axes[0])
# axes[0].legend(loc = 'upper right')
# data[['sjtu_per','ssc_per']].plot(kind='line',style = '--.',alpha = 0.8,ax=axes[1])
# axes[1].legend(loc = 'upper right')
# plt.show()

# df.Wait_Time.hist(grid=False, bins=20, color='lightblue')
# plt.title('等待时间直方图')
# plt.xlabel('Job', size=10)
# plt.ylabel('等待时间', size=10)
# plt.ticklabel_format(axis='x', style="sci", scilimits=(0, 0))
# plt.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
# x_major_locator = MultipleLocator(40000)
# # 把x轴的刻度间隔设置为50000，并存在变量里
# y_major_locator = MultipleLocator(10000)
# # 把y轴的刻度间隔设置为10000，并存在变量里
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# # 把x轴的主刻度设置为20000的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10000的倍数

# print(df4.isna().values.any())
# 获取特征值
# X = df4.iloc[:, :-1].values
# print(X.shape)
# 获取标签值
# Y = df4.iloc[:, [11]].values
# print(Y)
# 使用sklearn 的DecisionTreeClassifier判断变量重要性
# 建立分类决策树模型对象
# dt_model = DecisionTreeClassifier(random_state=1)
# 将数据集的维度和目标变量输入模型
# dt_model.fit(X, Y)
# 获取所有变量的重要性
# feature_importance = dt_model.feature_importances_
# print("重要性：")
# print(feature_importance)
# 相关性分析
# print(df4.corr().fillna(-1))
# print(df4.corr()['Job_Number'].abs().sort_values(ascending=False))
# 特征之间相关性热力图
# plt.figure(figsize=(12,10))
# sns.heatmap(df4.corr().fillna(0.00001), annot=True, fmt='.2f', cmap='Reds')
# 多变量研究
# sns.pairplot(df4[["Submit_Time","Wait_Time","Run_Time","NAP"]])

# plt.scatter(df['RNP'], df['NAP'],  # 按照经纬度显示
#             s=df['Wait_Time'],  # 按照单价显示大小
#             c=df['Run_Time'],  # 按照总价显示颜色
#             alpha=0.4, cmap='Reds')
# plt.grid()
# plt.style.use('ggplot')

# 设置中文和负号正常显示
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False

# # 绘图：整体乘客的年龄箱线图
# plt.boxplot(x = np.log(df.Run_Time), # 指定绘图数据
#             patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
#             showmeans=True, # 以点的形式显示均值
#             boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
#             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
#             meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
#             medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
# # 设置y轴的范围
# # plt.ylim(0,170000)
#
# # 去除箱线图的上边框与右边框的刻度标签
# plt.tick_params(top='off', right='off')

# 箱线图
# box_1, box_2, box_3, box_4, box_5, box_6, box_7, box_8 = np.log(sjtu_2019_1['Run_Time']), np.log(
#     ssc_2019_1['Run_Time']), np.log(ssc_2018_1['Run_Time']), np.log(ssc_2017_1['Run_Time']), np.log(
#     tc4060_2019_1['Run_Time']), np.log(tc4060_2017_3['Run_Time']), np.log(tc4060_2018['Run_Time']), np.log(
#     wxsc_2017_1['Run_Time'])
# # box_1, box_2, box_3, box_4, box_5 = np.log(df['Run_Time'] / 360), np.log(df1['Run_Time'] / 360), np.log(df2['Run_Time'] / 360), np.log(df3['Run_Time'] / 360), np.log(df4['Run_Time'] / 360)
# plt.figure(figsize=(10, 5))
# plt.title('各超算中心运行时间箱型图')
# labels = 'sjtu_2019', 'ssc_2019', 'ssc_2018', 'ssc_2017', 'tc4060_2019', 'tc4060_2017', 'tc4060_2018', 'wxsc_2017'
# plt.boxplot([box_1, box_2, box_3, box_4, box_5, box_6, box_7, box_8], labels=labels, patch_artist=True,
#             # 要求用自定义颜色填充盒形图，默认白色填充
#             showmeans=True,  # 以点的形式显示均值
#             boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色
#             flierprops={'marker': 'o', 'markerfacecolor': 'r','markersize':6},  # 设置异常值属性，点的形状、填充色和边框色
#             meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色
#             medianprops={'linestyle': '--', 'color': 'red'})
# plt.show()

# 离散化 分箱
# pd.cut(x, bins, right)：按照组数对x分组，且返回一个和x同样长度的分组dataframe，right → 是否右边包含，默认True
# 通过groupby查看不同组的数据频率分布
# 给源数据data添加“分组区间”列
# key = 'Run_Time'
# run_cut = pd.cut(sjtu_2019[key], 20, right=False)
# run_cut_count = run_cut.value_counts(sort=False)

# sjtu_2019['%s分组区间' % key] = run_cut.values
# print(run_cut.head(), '\n=================================')
# print(run_cut)
# print(sjtu_2019.head())
