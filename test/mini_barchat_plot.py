import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.axisartist as axisartist

# 数据集
accu1, accu2 = 68.04, 70.28
# accu1, accu2 = 92.05, 92.62
# 设置画布风格
plt.style.use('seaborn-white')
# plt.rcParams['figure.figsize']=(4, 5)
# # plt.style.use('seaborn-darkgrid')
# fig, ax = plt.subplots(dpi=600)

#创建画布
fig = plt.figure(figsize=(4, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)

# fig.suptitle('Source dataset accuracy', fontsize=15)
# ax.set_title('Source dataset accuracy', fontsize=15)
ax.set_title('Target dataset accuracy', fontsize=15)

N, width = 2, 0.05
ind = np.array([0.15])
# 0, 0.1, 0.2, 0.3, 0.4, 0.5
# 绘制柱形图
rects1 = ax.bar(ind, accu1, width=width, color='#DF8344', label='Conventional method')
# rects2 = ax.bar(ind + width, accu2, width, color='#7EAB55', label='Ada-Con')
rects2 = ax.bar(ind + width, accu2, width, color='#7EAB55', label='CES-JC')

# 设置柱形图x轴和y轴长度
ax.set_ylim(60, 73.5); ax.set_xlim(0, 0.35)
# ax.set_ylim(88, 93.5); ax.set_xlim(0, 0.35)

# 绘制表格横线
ax.set_ylabel("Accuracy(%)", fontsize=15)
# ax.set_ylabel("Accuracy(%)")

# 设置纵坐标间隔
y_major_locator = MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

# 绘制文本
ax.set_xticklabels("", fontsize=10)
# ax.set_yticklabels("", fontsize=10)
ax.set_yticklabels("")

ax.axis['left'].set_axisline_style("-|>",size=1.5)
ax.axis["bottom"].set_axisline_style("-|>",size=1.5)
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# 设置图例
# ax.legend((rects1[0], rects2[0]), ('normal pruned', 'graft pruned'), loc=2)
# ax.legend(loc=2, frameon=True, fontsize=8)
ax.legend(loc=2, frameon=True)

# plt.savefig('mini_accuracy.jpeg', dpi=600)
# plt.savefig('mini_accuracy2.jpeg', dpi=600)
# plt.savefig('mini_accuracy3.jpeg', dpi=600)
plt.savefig('mini_accuracy4.jpeg', dpi=600)
plt.show()