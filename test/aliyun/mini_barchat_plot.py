import numpy as np
from matplotlib.pyplot import MultipleLocator
import mpl_toolkits.axisartist as axisartist
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据集
# accu1, accu2 = 68.04, 70.28
# accu1, accu2 = 92.05, 92.62
# accu1, accu2 = 92189, 86092
accu1, accu2 = 10, 7
# 设置画布风格
plt.style.use('seaborn-white')

#创建画布
fig = plt.figure(figsize=(4, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)

ax.set_title('maxmemory=1G, Tairv4 以不同方式\n'
             '插入10KB大小的Key', fontsize=10)

N, width = 2, 0.05
ind = np.array([0.15])
# 0, 0.1, 0.2, 0.3, 0.4, 0.5
# 绘制柱形图
rects1 = ax.bar(ind, accu1, width=width, color='#404040', label='直接插入')
plt.text(0.126, 10.3, '92189', fontsize=10, color='#404040')
rects2 = ax.bar(ind + width, accu2, width, color='#AFABAB', label='先插入并删除64字节的key,\n然后插入10KB的key')
plt.text(0.178, 7.3, '86092', fontsize=10, color='#404040')


# 设置柱形图x轴和y轴长度
# ax.set_ylim(60, 73.5); ax.set_xlim(0, 0.35)
ax.set_ylim(0, 15); ax.set_xlim(0, 0.35)

# 绘制表格横线
# ax.set_ylabel("Accuracy(%)", fontsize=15)
ax.set_ylabel("10KB key 的数量", fontsize=15)
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
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.savefig('ali.jpeg', dpi=600)
# plt.savefig('mini_accuracy2.jpeg', dpi=600)
plt.show()