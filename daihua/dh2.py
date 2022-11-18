import matplotlib.pyplot as plt
import numpy as np

# y1 = [0.7325, 0.8565, 0.6573, 0.5596, 0.6467, 0.7082, 0.7340, 0.7989, 0.7765, 0.8342]
# y2 = [0.7720, 0.8237, 0.537,  0.5108, 0.5238, 0.5376, 0.6459, 0.7623, 0.7429, 0.8193]
# y3 = [0.3251, 0.5095, 0.4034, 0.384,  0.4434, 0.3584, 0.3329, 0.3041, 0.4207, 0.4811]

y1 = [0.7712, 0.8035, 0.5537, 0.4881, 0.5498, 0.5687, 0.6211, 0.7306, 0.7029, 0.7728]
y2 = [0.6835, 0.7227, 0.3914, 0.3461, 0.3767, 0.3874, 0.4977, 0.6363, 0.6302, 0.716]
y3 = [0.1463, 0.211,  0.1977, 0.1939, 0.2284, 0.1835, 0.2146, 0.2005, 0.1838, 0.2522]

x_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# x_labels = ['auto', 'dog', 'bird', 'cat', 'deer', 'frog', 'horse',
#             'ship', 'truck', 'airplane']
# 设置画布风格
plt.style.use('seaborn-darkgrid')
# 设置分辨率
# plt.rcParams['']
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
fig, ax = plt.subplots(dpi=600)

# 柱状图绘制位置及柱状图宽度
ind = list(range(1, 41, 4)); width = 1
print(ind)
ind2 = [item + width for item in ind]
ind3 = [item + width for item in ind2]
xticks_ind = [item + width / 2 for item in ind]
# 绘制柱状图
rects1 = ax.bar(ind, y1, width=width, color='green', label='Adversarial Accuracy')
# rects1 = ax.bar(ind, y1, width=width, color='#DF8344', label='True Adversarial Samples')
# #f10c45
rects2 = ax.bar(ind2, y2, width, color='orange', label='Clean Accuracy')
# rects2 = ax.bar(ind2, y2, width, color='#7EAB55', label='All Samples')

rects3 = ax.bar(ind3, y3, width, color='#fc5a50', label='False Adversarial Samples')
# rects3 = ax.bar(ind3, y3, width, color='#F5C242', label='False Adversarial Samples')

ax.set_ylim(0, 1.10)

# 绘制x轴和y轴坐标
ax.set_axisbelow(True)
ax.set_xticks(xticks_ind)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlabel('Class', fontsize=10)

# ax.legend()
ax.legend(loc=2, frameon=True)
plt.savefig('image4.jpeg', dpi=600)
plt.show()
