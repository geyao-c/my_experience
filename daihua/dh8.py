import matplotlib.pyplot as plt
import numpy as np

y1 = [89.60, 91.40, 87.10, 87.70, 87.60, 87.90, 86.50, 88.50, 88.40, 91.80]
y2 = [67.90, 82.50, 33.60, 17.90, 20.60, 39.20, 45.90, 69.90, 75.90, 76.40]

# x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 设置画布风格
plt.style.use('seaborn-darkgrid')
# 设置分辨率
# plt.rcParams['']
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
fig, ax = plt.subplots(dpi=600)

# 柱状图绘制位置及柱状图宽度
ind = list(range(1, 31, 3)); width = 1
ind2 = [item + width for item in ind]
xticks_ind = [item + width / 2 for item in ind]
# 绘制柱状图
# rects1 = ax.bar(ind, y1, width=width, color='green', label='BTRA(ours)')
rects1 = ax.bar(ind, y1, width=width, color='#DF8344', label='BTRA(ours)')
# #f10c45
# rects2 = ax.bar(ind2, y2, width, color='orange', label='SCORE')
rects2 = ax.bar(ind2, y2, width, color='#7EAB55', label='SCORE')

ax.set_ylim(0, 110)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
# ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_ylabel("The proportion of the sample(%)", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image11.jpeg', dpi=600)
plt.show()
