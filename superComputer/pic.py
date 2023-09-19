import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, Locator
import pandas as pd

# x = ["id", "Submit_Time", "Wait_Time", "Run_Time", "NAP", "RNP", "User_ID", "Queue_Number"]
# y = [339201681, 828210292, 182352322, 407366457, 98230044, 98230044, 20160504, 15931294]
# fig, ax = plt.subplots(figsize=(10, 7))
# # 设置刻度字体大小
# plt.xticks(fontsize=15, rotation=15, weight='bold')
# plt.yticks(fontsize=15, weight='bold')
# ax.set_ylabel('Scores', fontsize=15, weight='bold')
# ax.set_xlabel('Features', fontsize=15, weight='bold')
# ax.bar(x=x, height=y, width=0.5)
# ax.set_title("SelectKBest", fontsize=20, weight='bold')
# plt.savefig('E:\paper\\2203\\figures\skb.png', bbox_inches='tight', dpi=128)
#
# x2 = ["id", "Submit_Time", "Wait_Time", "Run_Time", "NAP", "RNP", "User_ID", "Queue_Number"]
# y2 = [-19124.564, -12148.496, 2494.202, -5203.794, 19331.822, 19331.822, 19220.976, 19354.746]
# fig2, ax = plt.subplots(figsize=(10, 7))
# # 设置刻度字体大小
# plt.xticks(fontsize=15, rotation=15, weight='bold')
# plt.yticks(fontsize=15, weight='bold')
# ax.set_ylabel('Scores', fontsize=15, weight='bold')
# ax.set_xlabel('Features', fontsize=15, weight='bold')
# ax.axhline(0, color='gray', linewidth=0.8)
# ax.bar(x=x2, height=y2, width=0.5)
# ax.set_title("ReliefF", fontsize=20, weight='bold')
# plt.savefig('E:\paper\\2203\\figures\\rrff.png', bbox_inches='tight', dpi=128)
# plt.show()

# 预测结果折线图
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plt.figure(figsize=(20, 9), dpi=200)
# # 真实值
# data01 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBtrueNoCV.csv", nrows=300)
# data02 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\RFtrueNoCV.csv", nrows=300)
# data03 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\DTtrueNoCV.csv", nrows=300)
# data04 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\KNNtrueNoCV.csv", nrows=300)
# data05 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\SVMtrueNoCV.csv", nrows=300)
# data06 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\GDBTtrueNoCV.csv", nrows=300)
#
# # 预测值
# data1 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBpredictionNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\RFpredictionNoCV.csv", nrows=300)
# data3 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\DTpredictionNoCV.csv", nrows=300)
# data4 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\KNNpredictionNoCV.csv", nrows=300)
# data5 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\SVMpredictionNoCV.csv", nrows=300)
# data6 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\GDBTpredictionNoCV.csv", nrows=300)

# # ReliefF
# # 真实值
# data01 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\LGBtrueNoCV.csv", nrows=300)
# data02 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\RFtrueNoCV.csv", nrows=300)
# data03 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\DTtrueNoCV.csv", nrows=300)
# data04 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\KNNtrueNoCV.csv", nrows=300)
# data05 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\SVMtrueNoCV.csv", nrows=300)
# data06 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\GDBTtrueNoCV.csv", nrows=300)
# #
# # 预测值
# data1 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\LGBpredictionNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\RFpredictionNoCV.csv", nrows=300)
# data3 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\DTpredictionNoCV.csv", nrows=300)
# data4 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\KNNpredictionNoCV.csv", nrows=300)
# data5 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\SVMpredictionNoCV.csv", nrows=300)
# data6 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\GDBTpredictionNoCV.csv", nrows=300)

# # 全特征
# # 真实值
# data01 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\LGBtrueNoCV.csv", nrows=300)
# data02 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\RFtrueNoCV.csv", nrows=300)
# data03 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\DTtrueNoCV.csv", nrows=300)
# data04 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\KNNtrueNoCV.csv", nrows=300)
# data05 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\SVMtrueNoCV.csv", nrows=300)
# data06 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\GDBTtrueNoCV.csv", nrows=300)
# #
# # 预测值
# data1 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\LGBpredictionNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\RFpredictionNoCV.csv", nrows=300)
# data3 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\DTpredictionNoCV.csv", nrows=300)
# data4 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\KNNpredictionNoCV.csv", nrows=300)
# data5 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\SVMpredictionNoCV.csv", nrows=300)
# data6 = pd.read_csv("D:\pythonProject\spark\datas\\result\\full\GDBTpredictionNoCV.csv", nrows=300)
#
# fig, axs = plt.subplots(2, 3, figsize=(30, 10), dpi=200)
# axs[0, 0].plot(data01, color='red', label="Label")
# axs[0, 0].plot(data1, color='blue', label="Prediction")
# axs[0, 0].set_ylabel('Status')
# axs[0, 0].set_title('LGBM')
# axs[0, 0].legend(loc=0)
# axs[0, 1].plot(data02, color='red', label="Label")
# axs[0, 1].plot(data2, color='blue', label="Prediction")
# axs[0, 1].set_title('Random Forest')
# # axs[0, 1].legend(loc=0)
# axs[0, 2].plot(data03, color='red', label="Label")
# axs[0, 2].plot(data3, color='blue', label="Prediction")
# axs[0, 2].set_title('DecisionTree')
# # axs[0, 2].legend(loc=0)
# axs[1, 0].plot(data04, color='red', label="Label")
# axs[1, 0].plot(data4, color='blue', label="Prediction")
# axs[1, 0].set_ylabel('Status')
# axs[1, 0].set_title('KNN')
# # axs[1, 0].legend(loc=0)
# axs[1, 1].plot(data05, color='red', label="Label")
# axs[1, 1].plot(data5, color='blue', label="Prediction")
# axs[1, 1].set_title('SVM')
# # axs[1, 1].legend(loc=0)
# axs[1, 2].plot(data06, color='red', label="Label")
# axs[1, 2].plot(data6, color='blue', label="Prediction")
# axs[1, 2].set_title('GBDT')
# axs[1, 2].legend(loc=0)
# plt.savefig('E:\paper\\2203\\figures\\SKBall.png', bbox_inches='tight', dpi=200)
# plt.savefig('E:\paper\\2203\\figures\\REFall.png', bbox_inches='tight', dpi=200)
# plt.savefig('E:\paper\\2203\\figures\\Fullall.png', bbox_inches='tight', dpi=200)

# 单独的图
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBpredictionNoCV.csv", nrows=300)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\RFpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\DTpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\KNNpredictionNoCV.csv",nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\SVMpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\GDBTpredictionNoCV.csv", nrows=400)
#
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBtrueNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\RFtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\DTtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\KNNtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\SVMtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\GDBTtrueNoCV.csv", nrows=400)
#

# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\LGBpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\RFpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\DTpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\KNNpredictionNoCV.csv",nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\SVMpredictionNoCV.csv", nrows=400)
# data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\GDBTpredictionNoCV.csv", nrows=400)
#
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\reliefF\LGBtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\RFtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\DTtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\KNNtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\SVMtrueNoCV.csv", nrows=400)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\GDBTtrueNoCV.csv", nrows=400)
#
# 画在一起的单独图
# data1 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\LGBpredictionNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\RFpredictionNoCV.csv", nrows=300)
# data3 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\DTpredictionNoCV.csv", nrows=300)
# data4 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\KNNpredictionNoCV.csv", nrows=300)
# data5 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\SVMpredictionNoCV.csv", nrows=300)
# data6 = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\GDBTpredictionNoCV.csv", nrows=300)
# label_data = pd.read_csv("D:\pythonProject\spark\datas\\result\\reliefF\LGBtrueNoCV.csv", nrows=300)
#
# data1 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBpredictionNoCV.csv", nrows=300)
# data2 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\RFpredictionNoCV.csv", nrows=300)
# data3 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\DTpredictionNoCV.csv", nrows=300)
# data4 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\KNNpredictionNoCV.csv",nrows=300)
# data5 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\SVMpredictionNoCV.csv", nrows=300)
# data6 = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\GDBTpredictionNoCV.csv", nrows=300)
# label_data = pd.read_csv("D:\pythonProject\spark\datas\\result\SKB\LGBtrueNoCV.csv", nrows=300)
# x = np.arange(0, 301, step=10)
# y = np.arange(0, 6, step=0.5)
# plt.figure(figsize=(20, 9), dpi=200)
# plt.plot(data2, color='blue', label="Label")
# plt.plot(data, color='red', label="Prediction")


# plt.plot(data1, color='fuchsia', label="lgb")
# plt.plot(data2, color='yellow', label="RF")
# plt.plot(data3, color='grey', label="DT")
# plt.plot(data4, color='m', label="KNN")
# plt.plot(data5, color='red', label="SVM")
# plt.plot(data6, color='g', label="GDBT")
# plt.plot(label_data, color='blue', marker='*', label="Label")
# plt.xticks(x)
# plt.yticks(y)
# plt.legend(loc=0)
# # # plt.xticks(x)
# plt.title("LightGBM")
# # # plt.title("Random Forest")
# # # plt.title("DecisionTree")
# # # plt.title("KNN")
# # # plt.title("SVM")
# # plt.title("GDBT")
# plt.ylabel("Status")
# # plt.savefig('E:\paper\\2203\\figures\\LGB.png', bbox_inches='tight', dpi=128)
# # plt.savefig('E:\paper\\2203\\figures\\RF.png', bbox_inches='tight', dpi=128)
# # plt.savefig('E:\paper\\2203\\figures\\DT.png', bbox_inches='tight', dpi=128)
# # plt.savefig('E:\paper\\2203\\figures\\KNN.png', bbox_inches='tight', dpi=128)
# # plt.savefig('E:\paper\\2203\\figures\\SVM.png', bbox_inches='tight', dpi=128)
# plt.savefig('E:\paper\\2203\\figures\\GDBT.png', bbox_inches='tight', dpi=128)
# plt.savefig('E:\paper\\2203\\figures\\allREF.png', bbox_inches='tight', dpi=128)
# plt.savefig('E:\paper\\2203\\figures\\allSKB.png', bbox_inches='tight', dpi=128)

# 对比柱状图
# from pylab import mpl
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# labels = ['ACC', 'Precision', 'Recall', 'F-score']
#
# # labels = ['BO(TPE)', 'Random Search', 'Grid Search']
# labels = ['TPE贝叶斯', '随机搜索', '网格搜索']

# SJTU 贝叶斯调参
# auc = [0.9592, 0.9065, 0.8834]
# aucc = [0.9647, 0.9191738462436064, 0.8934635057080068]
# pre = [0.9636, 0.918771227240777, 0.8944759668685239]
# rec = [0.9647, 0.9191738462436064, 0.8934635057080068]
# f = [0.9641, 0.9178850186107028, 0.8883358806694086]
# y = [11.25, 23.4, 67.35]

#USTC 贝叶斯调参
# auc = [0.9081, 0.8665, 0.8434]
# aucc = [0.9589, 0.9021738462436064, 0.8934592154080068]
# pre = [0.9547, 0.892771227240777, 0.8944759668685239]
# rec = [0.9589, 0.9021738462436064, 0.8934592154080068]
# f = [0.9568, 0.8974479094582101, 0.8939673020384302]
# y = [10.37, 21.44, 62.55]

#SSC 贝叶斯调参
# auc = [0.9443, 0.8865, 0.8434]
# aucc = [0.9566, 0.9065412462436064, 0.9075612154080068]
# pre = [0.9496, 0.912771227240777, 0.9044458968685239]
# rec = [0.9566, 0.9065412462436064, 0.9075612154080068]
# f = [0.9531, 0.9096455698930961, 0.9060008781108071]
# y = [9.87, 18.44, 59.68]

#WXSC 贝叶斯调参
# auc = [0.9618, 0.9144, 0.8789]
# aucc = [0.9391, 0.8942134662436064, 0.8845262154080068]
# pre = [0.9387, 0.909764327240777, 0.89645158968685239]
# rec = [0.9391, 0.8942134662436064, 0.8845262154080068]
# f = [0.9389, 0.9019218700640844, 0.8904489765721356]
# y = [7.39, 12.84, 45.68]
# x = np.arange(len(labels))  # the label locations
# width = 0.1  # the width of the bars
# fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
# ax2 = plt.twinx()
# # ax2.set_ylabel("Running Time/(min)")
# ax2.set_ylabel("运行时间/(min)")
# # plt.plot(y, "red", marker="o", linestyle="dashed", label="Running time")
# plt.plot(y, "red", marker="o", linestyle="dashed", label="运行时间")
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='right', va='bottom', fontsize=11)
#
# rects0 = ax.bar(x - width - 0.12, auc, width, alpha=0.5, label='AUC')
# rects1 = ax.bar(x - width -0.01, aucc, width, alpha=0.5, label='Accuracy')
# rects2 = ax.bar(x - width + 0.10, pre, width, alpha=0.5, label='Precision')
# rects3 = ax.bar(x + 0.11, rec, width, alpha=0.5, label='Recall')
# rects4 = ax.bar(x + width + 0.12, f, width, alpha=0.5, label='F-score')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Evaluation indices')
# ax.set_ylabel('评估指标')
# # ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend(loc=6)
# plt.legend(loc=7)
# fig.tight_layout()
# plt.savefig('C:\\Users\\stone\\Desktop\\dlw\\BOwxsc.png', bbox_inches='tight', dpi=200)
# plt.show()
#
# rects1 = ax[0, 0].bar(x - width / 2, knn_pre, width, label='Before')
# rects2 = ax[0, 0].bar(x + width / 2, knn_after, width, label='After')
# rects3 = ax[0, 1].bar(x - width / 2, dt_pre, width, label='Before')
# rects4 = ax[0, 1].bar(x + width / 2, dt_after, width, label='After')
# rects5 = ax[0, 2].bar(x - width / 2, rf_pre, width, label='Before')
# rects6 = ax[0, 2].bar(x + width / 2, rf_after, width, label='After')
# rects7 = ax[1, 0].bar(x - width / 2, gbdt_pre, width, label='Before')
# rects8 = ax[1, 0].bar(x + width / 2, gbdt_after, width, label='After')
# rects9 = ax[1, 1].bar(x - width / 2, lgbm_pre, width, label='Before')
# rects10 = ax[1, 1].bar(x + width / 2, lgbm_after, width, label='After')
# rects11 = ax[1, 2].bar(x - width / 2, svm_pre, width, label='Before')
# rects12 = ax[1, 2].bar(x + width / 2, svm_after, width, label='After')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[0, 0].set_ylabel('Score', fontsize=14)
# ax[0, 0].set_title('KNN', fontsize=15)
# ax[0, 0].set_xticks(x, labels)
# # ax[0, 0].set_ylim(0.88, 0.92)
# ax[0, 0].set_ylim(0.94, 0.955)
# ax[0, 0].legend()
# ax[0, 0].bar_label(rects1, padding=3)
# ax[0, 0].bar_label(rects2, padding=3)
# fig.tight_layout()
# plt.savefig('E:\paper\\2203\\figures\\SKBVSall.png', bbox_inches='tight', dpi=200)
# plt.savefig('E:\paper\\2203\\figures\\REFVSall.png', bbox_inches='tight', dpi=200)
# SKB
# ylabels = ['ACC', 'PRE', 'Recall', 'F-score']
# knn_pre = [0.904957311, 0.901591356, 0.904957311, 0.901926001]
# dt_pre = [0.934635057, 0.934264566, 0.934635057, 0.934438949]
# rf_pre = [0.948112565, 0.947194677, 0.948112565, 0.946950068]
# gbdt_pre = [0.927906027, 0.928035356, 0.927906027, 0.924670919]
# lgbm_pre = [0.898733931, 0.900659288, 0.898733931, 0.896389612]
# svm_pre = [0.740562827,
#            0.726380099,
#            0.740562827,
#            0.733423311
#            ]
#
# knn_after = [0.910266633, 0.907457808, 0.910266633, 0.906921733]
# dt_after = [0.936327039, 0.935491477, 0.936327039, 0.935772086]
# rf_after = [0.949590618,
#             0.948702585,
#             0.949590618,
#             0.948366947
#             ]
# gbdt_after = [0.937668955,
#               0.937193993,
#               0.937668955,
#               0.935560868
#               ]
# lgbm_after = [0.923996966,
#               0.922816472,
#               0.923996966,
#               0.920716219
#               ]
# svm_after = [0.780854567,
#              0.779852909,
#              0.780854567,
#              0.78034968
#              ]

# REF
# knn_pre = [0.947743052,
#            0.946867326,
#            0.947743052,
#            0.947163259
#            ]
# dt_pre = [0.951243704,
#           0.951097003,
#           0.951243704,
#           0.951159033
#           ]
# rf_pre = [0.959003481,
#           0.958407944,
#           0.959003481,
#           0.95832956
#           ]
# gbdt_pre = [0.937980124,
#             0.936693644,
#             0.937980124,
#             0.935467464
#             ]
# lgbm_pre = [0.915809331,
#             0.915777296,
#             0.915809331,
#             0.913885337
#             ]
# svm_pre = [0.774821953,
#            0.762649413,
#            0.774821953,
#            0.768953194
#
#            ]
#
# knn_after = [0.952332795,
#              0.951614844,
#              0.952332795,
#              0.951673906
#              ]
# dt_after = [0.952780101,
#             0.952760352,
#             0.952780101,
#             0.952760048
#             ]
# rf_after = [0.959567475,
#             0.958946984,
#             0.959567475,
#             0.958997601
#
#             ]
# gbdt_after = [0.961784554,
#               0.961192062,
#               0.961784554,
#               0.961277572
#
#               ]
# lgbm_after = [0.943386686,
#               0.942311762,
#               0.943386686,
#               0.941985983
#
#               ]
# svm_after = [0.816354011,
#              0.802564853,
#              0.816354011,
#              0.80939627
#
#              ]
# x = np.arange(len(labels))  # the label locations
#
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots(2, 3, figsize=(25, 10), dpi=200)
#
# rects1 = ax[0, 0].bar(x - width / 2, knn_pre, width, label='Before')
# rects2 = ax[0, 0].bar(x + width / 2, knn_after, width, label='After')
# rects3 = ax[0, 1].bar(x - width / 2, dt_pre, width, label='Before')
# rects4 = ax[0, 1].bar(x + width / 2, dt_after, width, label='After')
# rects5 = ax[0, 2].bar(x - width / 2, rf_pre, width, label='Before')
# rects6 = ax[0, 2].bar(x + width / 2, rf_after, width, label='After')
# rects7 = ax[1, 0].bar(x - width / 2, gbdt_pre, width, label='Before')
# rects8 = ax[1, 0].bar(x + width / 2, gbdt_after, width, label='After')
# rects9 = ax[1, 1].bar(x - width / 2, lgbm_pre, width, label='Before')
# rects10 = ax[1, 1].bar(x + width / 2, lgbm_after, width, label='After')
# rects11 = ax[1, 2].bar(x - width / 2, svm_pre, width, label='Before')
# rects12 = ax[1, 2].bar(x + width / 2, svm_after, width, label='After')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[0, 0].set_ylabel('Score', fontsize=14)
# ax[0, 0].set_title('KNN', fontsize=15)
# ax[0, 0].set_xticks(x, labels)
# # ax[0, 0].set_ylim(0.88, 0.92)
# ax[0, 0].set_ylim(0.94, 0.955)
# ax[0, 0].legend()
# ax[0, 0].bar_label(rects1, padding=3)
# ax[0, 0].bar_label(rects2, padding=3)
#
# ax[0, 1].set_title('DecisionTree', fontsize=15)
# ax[0, 1].set_xticks(x, labels)
# # ax[0, 1].set_ylim(0.93, 0.94)
# ax[0, 1].set_ylim(0.95, 0.955)
# ax[0, 1].legend()
# ax[0, 1].bar_label(rects3, padding=3)
# ax[0, 1].bar_label(rects4, padding=3)
#
# ax[0, 2].set_title('RandomForest', fontsize=15)
# ax[0, 2].set_xticks(x, labels)
# # ax[0, 2].set_ylim(0.94, 0.96)
# ax[0, 2].set_ylim(0.955, 0.96)
# ax[0, 2].legend()
# ax[0, 2].bar_label(rects5, padding=3)
# ax[0, 2].bar_label(rects6, padding=3)
#
# ax[1, 0].set_ylabel('Score', fontsize=14)
# ax[1, 0].set_title('GBDT', fontsize=15)
# ax[1, 0].set_xticks(x, labels)
# # ax[1, 0].set_ylim(0.92, 0.94)
# ax[1, 0].set_ylim(0.935, 0.965)
# ax[1, 0].legend()
# ax[1, 0].bar_label(rects7, padding=3)
# ax[1, 0].bar_label(rects8, padding=3)
#
# ax[1, 1].set_title('LightGBM', fontsize=15)
# ax[1, 1].set_xticks(x, labels)
# # ax[1, 1].set_ylim(0.88, 0.93)
# ax[1, 1].set_ylim(0.91, 0.95)
# ax[1, 1].legend()
# ax[1, 1].bar_label(rects9, padding=3)
# ax[1, 1].bar_label(rects10, padding=3)
#
# ax[1, 2].set_title('SVM', fontsize=15)
# ax[1, 2].set_xticks(x, labels)
# # ax[1, 2].set_ylim(0.72, 0.79)
# ax[1, 2].set_ylim(0.76, 0.82)
# ax[1, 2].legend()
# ax[1, 2].bar_label(rects11, padding=3)
# ax[1, 2].bar_label(rects12, padding=3)
# fig.tight_layout()
# plt.savefig('E:\paper\\2203\\figures\\SKBVSall.png', bbox_inches='tight', dpi=200)
# plt.savefig('E:\paper\\2203\\figures\\REFVSall.png', bbox_inches='tight', dpi=200)

# # 运行时间对比图
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 6), dpi=200)
x = ["1", "8", "16", "32", "64"]
# SJTU
# y = [7.6, 12.54, 16.55, 24.55, 32.45]
# y2 = [6.55, 12.12, 17.20, 22.23, 29.12]
# y3 = [6.05, 9.02, 14.41, 19.88, 28.66]
# y4 = [5.23, 6.55, 14.65, 18.55, 24.55]
# y5 = [4.02, 5.75, 9.63, 11.52, 18.89]
# y6 = [2.25, 4.55, 8.12, 12.24, 15.65]
#
# # USTC
# y = [8.4, 13.64, 17.25, 25.50, 33.85]
# y2 = [7.85, 12.33, 17.20, 22.33, 29.02]
# y3 = [7.05, 11.02, 16.41, 19.92, 29.66]
# y4 = [4.65, 6.55, 13.65, 18.65, 24.70]
# y5 = [2.77, 6.456, 9.88, 13.92, 20.89]
# y6 = [2.63, 4.55, 9.12, 11.64, 17.65]

# # SSC
# y = [6.4, 10.64, 14.25, 20.60, 29.85]
# y2 = [6.05, 9.73, 15.20, 18.98, 27.02]
# y3 = [5.15, 10.02, 13.02, 18.00, 25.42]
# y4 = [3.05, 6.15, 10.95, 11.65, 21.30]
# y5 = [2.37, 5.32, 8.38, 11.02, 17.89]
# y6 = [2.02, 4.04, 7.12, 9.64, 15.35]

# WXSC
y = [4.90, 7.64, 10.15, 14.60, 20.85]
y2 = [4.55, 7.03, 10.00, 13.98, 20.02]
y3 = [4.15, 6.62, 9.02, 13.00, 18.72]
y4 = [2.75, 6.15, 9.25, 10.65, 15.60]
y5 = [2.37, 5.32, 7.88, 8.12, 13.79]
y6 = [1.89, 3.77, 6.12, 7.44, 12.35]

ax.plot(x, y, marker='o', label="x01-local")
ax.plot(x, y2, marker='^', label="x06-local")
ax.plot(x, y3, marker='*', label="x12-local")
ax.plot(x, y4, marker='p', label="x01-cluster")
ax.plot(x, y5, marker='x', label="x06-cluster")
ax.plot(x, y6, marker='d', label="x12-cluster")
ax.grid(axis='x')
ax.legend()

# ax.set_xlabel('Dataset size')
ax.set_xlabel('数据集大小倍数')
# ax.set_ylabel('Cost time(min)')
# ax.set_ylabel('Cost time/(s)')
ax.set_ylabel('预测时间/(秒)')
# ax.set_title('RS-BOLGBM local and cluster runtime comparison')
# ax.set_title('RS-BOLGBM 本地与集群运行时间对比')
# plt.savefig('E:\paper\\2203\\figures\\lvsc.png', bbox_inches='tight', dpi=200)
plt.savefig('C:\\Users\\stone\\Desktop\\dlw\\RTwxsc.png', bbox_inches='tight', dpi=200)
plt.show()
