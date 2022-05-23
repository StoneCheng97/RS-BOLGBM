import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
# y = iris.target
# print(y)
y_test0 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1GBDT,SKB调参",nrows=4000)
y_test1 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1SVM SGDClassifier",nrows=4200)
y_test2 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1DecisionTreeGini",nrows=3000)
y_test3 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1RandomForest",nrows=50000)
y_test4 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1KNN")
y_test5 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\\true_values_1LGBM无调参",nrows=4000)

y0 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1GBDT,SKB调参",nrows=4000)
y1 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1SVM SGDClassifier",nrows=4200)
y2 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1DecisionTreeGini",nrows=3000)
y3 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1RandomForest",nrows=50000)
y4 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1KNN")
y5 = pd.read_csv("D:\pythonProject\spark\superComputer\\results\\full\predict_values_1LGBM无调参",nrows=4000)

y_test0 = np.array(y_test0)
y_test1 = np.array(y_test1)
y_test2 = np.array(y_test2)
y_test3 = np.array(y_test3)
y_test4 = np.array(y_test4)
y_test5 = np.array(y_test5)
y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)
y5 = np.array(y5)
# Binarize the output
y0 = label_binarize(y0, classes=[0, 1, 2, 3, 4, 5])
y1 = label_binarize(y1, classes=[0, 1, 2, 3, 4, 5])
y2 = label_binarize(y2, classes=[0, 1, 2, 3, 4, 5])
y3 = label_binarize(y3, classes=[0, 1, 2, 3, 4, 5])
y4 = label_binarize(y4, classes=[0, 1, 2, 3, 4, 5])
y5 = label_binarize(y5, classes=[0, 1, 2, 3, 4, 5])
y_test0 = label_binarize(y_test0, classes=[0, 1, 2, 3, 4, 5])
y_test1 = label_binarize(y_test1, classes=[0, 1, 2, 3, 4, 5])
y_test2 = label_binarize(y_test2, classes=[0, 1, 2, 3, 4, 5])
y_test3 = label_binarize(y_test3, classes=[0, 1, 2, 3, 4, 5])
y_test4 = label_binarize(y_test4, classes=[0, 1, 2, 3, 4, 5])
y_test5 = label_binarize(y_test5, classes=[0, 1, 2, 3, 4, 5])
# print(y)
n_classes = y1.shape[1]
print(n_classes)
from sklearn.utils.multiclass import type_of_target

# print(type_of_target(y_test))
fpr0 = dict()
fpr1 = dict()
fpr2 = dict()
fpr3 = dict()
fpr4 = dict()
fpr5 = dict()
tpr0 = dict()
tpr1 = dict()
tpr2 = dict()
tpr3 = dict()
tpr4 = dict()
tpr5 = dict()
roc_auc0 = dict()
roc_auc1 = dict()
roc_auc2 = dict()
roc_auc3 = dict()
roc_auc4 = dict()
roc_auc5 = dict()
for i in range(n_classes):
    fpr0[i], tpr0[i], _ = roc_curve(y_test0[:, i], y0[:, i])
    roc_auc0[i] = auc(fpr0[i], tpr0[i])
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_test1[:, i], y1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_test2[:, i], y2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
for i in range(n_classes):
    fpr3[i], tpr3[i], _ = roc_curve(y_test3[:, i], y3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])
for i in range(n_classes):
    fpr4[i], tpr4[i], _ = roc_curve(y_test4[:, i], y4[:, i])
    roc_auc4[i] = auc(fpr4[i], tpr4[i])
for i in range(n_classes):
    fpr5[i], tpr5[i], _ = roc_curve(y_test5[:, i], y5[:, i])
    roc_auc5[i] = auc(fpr5[i], tpr5[i])

# plt.figure(figsize=(10, 7), dpi=200)
# lw = 2
# plt.plot(
#     fpr[3],
#     tpr[3],
#     color="darkorange",
#     lw=lw,
#     label="ROC curve (area = %0.2f)" % roc_auc[2],
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
# plt.legend(loc="lower right")
# plt.show()

# Compute micro-average ROC curve and ROC area
fpr0["micro"], tpr0["micro"], _ = roc_curve(y_test0.ravel(), y0.ravel())
fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test1.ravel(), y1.ravel())
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test2.ravel(), y2.ravel())
fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test3.ravel(), y3.ravel())
fpr4["micro"], tpr4["micro"], _ = roc_curve(y_test4.ravel(), y4.ravel())
fpr5["micro"], tpr5["micro"], _ = roc_curve(y_test5.ravel(), y5.ravel())
roc_auc0["micro"] = auc(fpr0["micro"], tpr0["micro"])
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])
roc_auc4["micro"] = auc(fpr4["micro"], tpr4["micro"])
roc_auc5["micro"] = auc(fpr5["micro"], tpr5["micro"])

# First aggregate all false positive rates
all_fpr0 = np.unique(np.concatenate([fpr0[i] for i in range(n_classes)]))
all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in range(n_classes)]))
all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
all_fpr3 = np.unique(np.concatenate([fpr3[i] for i in range(n_classes)]))
all_fpr4 = np.unique(np.concatenate([fpr4[i] for i in range(n_classes)]))
all_fpr5 = np.unique(np.concatenate([fpr5[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr0 = np.zeros_like(all_fpr0)
mean_tpr1 = np.zeros_like(all_fpr1)
mean_tpr2 = np.zeros_like(all_fpr2)
mean_tpr3 = np.zeros_like(all_fpr3)
mean_tpr4 = np.zeros_like(all_fpr4)
mean_tpr5 = np.zeros_like(all_fpr5)
for i in range(n_classes):
    mean_tpr0 += np.interp(all_fpr0, fpr0[i], tpr0[i])
for i in range(n_classes):
    mean_tpr1 += np.interp(all_fpr1, fpr1[i], tpr1[i])
for i in range(n_classes):
    mean_tpr2 += np.interp(all_fpr2, fpr2[i], tpr2[i])
for i in range(n_classes):
    mean_tpr3 += np.interp(all_fpr3, fpr3[i], tpr3[i])
for i in range(n_classes):
    mean_tpr4 += np.interp(all_fpr4, fpr4[i], tpr4[i])
for i in range(n_classes):
    mean_tpr5 += np.interp(all_fpr5, fpr5[i], tpr5[i])

# Finally average it and compute AUC
mean_tpr0 /= n_classes
mean_tpr1 /= n_classes
mean_tpr2 /= n_classes
mean_tpr3 /= n_classes
mean_tpr4 /= n_classes
mean_tpr5 /= n_classes

fpr0["macro"] = all_fpr0
fpr1["macro"] = all_fpr1
fpr2["macro"] = all_fpr2
fpr3["macro"] = all_fpr3
fpr4["macro"] = all_fpr4
fpr5["macro"] = all_fpr5
tpr0["macro"] = mean_tpr0
tpr1["macro"] = mean_tpr1
tpr2["macro"] = mean_tpr2
tpr3["macro"] = mean_tpr3
tpr4["macro"] = mean_tpr4
tpr5["macro"] = mean_tpr5
roc_auc0["macro"] = auc(fpr0["macro"], tpr0["macro"])
roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])
roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])
roc_auc3["macro"] = auc(fpr3["macro"], tpr3["macro"])
roc_auc4["macro"] = auc(fpr4["macro"], tpr4["macro"])
roc_auc5["macro"] = auc(fpr5["macro"], tpr5["macro"])
#
# macro_roc_auc_ovo = roc_auc_score(y_test0, y0, multi_class="ovo", average="macro")
# weighted_roc_auc_ovo = roc_auc_score(
#     y_test0, y0, multi_class="ovo", average="weighted"
# )
# macro_roc_auc_ovr = roc_auc_score(y_test0, y0, multi_class="ovr", average="macro")
# weighted_roc_auc_ovr = roc_auc_score(
#     y_test0, y0, multi_class="ovr", average="weighted"
# )
# print(
#     "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#     "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
# )
# print(
#     "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#     "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
# )
#
# # Plot all ROC curves
plt.figure(figsize=(10, 7), dpi=200)
lw = 2
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

plt.plot(
    fpr0["macro"],
    tpr0["macro"],
    label="GBDT (AUC = {0:0.4f})".format(roc_auc0["macro"]),
    color="yellow",
    # linestyle=":",
    linewidth=1.5,
)
plt.plot(
    fpr1["macro"],
    tpr1["macro"],
    label="SVM (AUC = {0:0.4f})".format(roc_auc1["macro"]),
    color="navy",
    # linestyle=":",
    linewidth=1.5,
)
plt.plot(
    fpr2["macro"],
    tpr2["macro"],
    label="DT (AUC = {0:0.4f})".format(roc_auc2["macro"]),
    color="red",
    # linestyle=":",
    linewidth=1.5,
)
plt.plot(
    fpr3["macro"],
    tpr3["macro"],
    label="RF (AUC = {0:0.4f})".format(roc_auc3["macro"]),
    color="cornflowerblue",
    # linestyle=":",
    linewidth=1.5,
)
plt.plot(
    fpr4["macro"],
    tpr4["macro"],
    label="KNN (AUC = {0:0.4f})".format(roc_auc4["macro"]),
    color="darkorange",
    # linestyle=":",
    linewidth=1.5,
)
plt.plot(
    fpr5["macro"],
    tpr5["macro"],
    label="RS-BOLGBM (AUC = {0:0.4f})".format(roc_auc5["macro"]),
    color="aqua",
    # linestyle=":",
    linewidth=1.5,
)

# colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "blue", "yellow"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of status {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves of RS-BOLGBM and other machine learning algorithms")
plt.legend(loc="lower right")
plt.savefig('E:\paper\\2203\\figures\\roc4.png', bbox_inches='tight', dpi=200)
plt.show()
