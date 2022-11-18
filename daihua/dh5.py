import matplotlib.pyplot as plt
import numpy as np

ay1 = [92.70, 95.70, 80.90, 78.30, 88.40, 79.00, 94.20, 93.30, 96.40, 93.60]
ay2 = [93.30, 95.70, 80.80, 75.40, 86.70, 79.20, 92.70, 92.60, 96.70, 93.00]

ry1 = [87.50, 90.60, 69.70, 68.10, 77.50, 68.10, 86.40, 85.20, 92.70, 88.90]
ry2 = [72.30, 81.80, 50.50, 38.90, 48.70, 52.90, 69.30, 75.30, 80.60, 79.10]

dy1 = [round(ay1[i] - ry1[i], 2) for i in range(len(ay1))]
dy2 = [round(ay2[i] - ry2[i], 2) for i in range(len(ay2))]

print(dy1)
print(dy2)
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
rects1 = ax.bar(ind, dy1, width=width, color='#DF8344', label='BTRA(ours)')
# #f10c45
# rects2 = ax.bar(ind2, y2, width, color='orange', label='SCORE')
rects2 = ax.bar(ind2, dy2, width, color='#7EAB55', label='SCORE')

ax.set_ylim(0, 45)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
# ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_ylabel("The accuracy-robustness gap(%)", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image8.jpeg', dpi=600)
plt.show()
