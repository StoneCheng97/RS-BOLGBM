import numpy as np
import matplotlib.pyplot as plt

x = ["id", "Submit_Time", "Wait_Time", "Run_Time", "NAP", "RNP", "User_ID", "Queue_Number"]
y = [339201681, 828210292, 182352322, 407366457, 98230044, 98230044, 20160504, 15931294]
fig, ax = plt.subplots(figsize=(10, 7))
# 设置刻度字体大小
plt.xticks(fontsize=15, rotation=15, weight='bold')
plt.yticks(fontsize=15, weight='bold')
ax.set_ylabel('Scores', fontsize=15, weight='bold')
ax.set_xlabel('Features', fontsize=15, weight='bold')
ax.bar(x=x, height=y, width=0.5)
ax.set_title("SelectKBest", fontsize=20, weight='bold')
plt.savefig('E:\paper\\2203\\figures\skb.png', dpi=128)

x2 = ["id", "Submit_Time", "Wait_Time", "Run_Time", "NAP", "RNP", "User_ID", "Queue_Number"]
y2 = [-19124.564, -12148.496, 2494.202, -5203.794, 19331.822, 19331.822, 19220.976, 19354.746]
fig2, ax = plt.subplots(figsize=(10, 7))
# 设置刻度字体大小
plt.xticks(fontsize=15, rotation=15, weight='bold')
plt.yticks(fontsize=15, weight='bold')
ax.set_ylabel('Scores', fontsize=15, weight='bold')
ax.set_xlabel('Features', fontsize=15, weight='bold')
ax.axhline(0, color='gray', linewidth=0.8)
ax.bar(x=x2, height=y2, width=0.5)
ax.set_title("ReliefF", fontsize=20, weight='bold')
plt.savefig('E:\paper\\2203\\figures\\rrff.png', dpi=128)
plt.show()
