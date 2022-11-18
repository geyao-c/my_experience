import matplotlib.pyplot as plt
import numpy as np

y1 = [5.61, 5.33, 13.97, 13.15, 12.33, 13.8, 8.28, 8.68, 3.94, 5.13]
y2 = [22.51, 14.52, 37.5, 48.41, 43.83, 33.21, 25.24, 18.68, 16.65, 14.95]

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

ax.set_ylim(0, 50)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
# ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_ylabel("The proportion of the sample(%)", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image9.jpeg', dpi=600)
plt.show()
