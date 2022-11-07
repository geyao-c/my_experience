import matplotlib.pyplot as plt
import numpy as np

y1 = [0.807, 0.502, 0.463, 0.349, 0.418, 0.639, 0.730, 0.780, 0.770, 0.704]
y2 = [0.957, 0.792, 0.808, 0.754, 0.867, 0.927, 0.926, 0.967, 0.930, 0.933]
# x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_labels = ['auto', 'dog', 'bird', 'cat', 'deer', 'frog', 'horse',
            'ship', 'truck', 'airplane']
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
rects1 = ax.bar(ind, y1, width=width, color='green', label='Adversarial Accuracy')
# rects1 = ax.bar(ind, y1, width=width, color='#DF8344', label='Adversarial Accuracy')
# #f10c45
rects2 = ax.bar(ind2, y2, width, color='orange', label='Clean Accuracy')
# rects2 = ax.bar(ind2, y2, width, color='#7EAB55', label='Clean Accuracy')

ax.set_ylim(0, 1.15)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image1.jpeg', dpi=600)
plt.show()
