import matplotlib.pyplot as plt
import numpy as np

y1 = [22.12, 12.57, 27.41, 26.19, 36.02, 23.68, 43.99, 20.93, 26.56, 19.92]
y2 = [96.02, 96.48, 96.13, 97.21, 97.57, 98.21, 99.13, 98.71, 97.76, 97.38]

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
rects1 = ax.bar(ind, y1, width=width, color='#DF8344', label='Highest - Second highest<0.5')
# #f10c45
# rects2 = ax.bar(ind2, y2, width, color='orange', label='SCORE')
rects2 = ax.bar(ind2, y2, width, color='#7EAB55', label='Highest - Second highest>0.5')

ax.set_ylim(0, 115)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
# ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_ylabel("Robustness(%)", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image10.jpeg', dpi=600)
plt.show()
