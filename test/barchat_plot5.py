import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

# 数据集
# pruned_normal = [71.88, 70.26]; pruned_graft = [72.29, 70.28]
# x_labels = ['42.8', ' ', '71.3']

pruned_normal = [94.14, 93.59]; pruned_graft = [94.12, 93.45]
x_labels = ['83.3', ' ', '87.3']

threshold = 93.99
# 设置画布风格
# plt.style.use('seaborn-white')
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(dpi=600)

# 设置柱形图x轴和y轴长度
# ax.set_ylim(66, 75)
ax.set_ylim(92, 95)
ax.set_xlim(0, 1.0)

# 设置纵坐标间隔
y_major_locator = MultipleLocator(1)
x_major_locator = MultipleLocator(0.2)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)

N = 2
ind = np.array([0.2, 0.7])
width = 0.1

# 绘制柱形图
rects1 = ax.bar(ind, pruned_normal, width=width, color='#DF8344', label='conventional pruning')
rects2 = ax.bar(ind + width, pruned_graft, width, color='#7EAB55', label='Ada-Con')


# 绘制表格横线
# ax.set_axisbelow(True)
# ax.grid(color='gray', axis='y', linewidth=1)
# xlb_ind = np.array([0.2 * i for i in range(0, 5)])
xlb_ind = np.array([0.2, 0.45, 0.7])
print(xlb_ind)
# xlb_ind = np.array([])
ax.set_xticks(xlb_ind + width/2)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_xlabel('Parameter pruned ratio(%)', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.get_xaxis().set_visible(False)
line = ax.plot([-0.3, 1.15], [threshold, threshold], "k--", color='r', label='baseline')
print(type(line))
# 绘制文本
def labelvalue(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.001 * height,
                '%.2f' % height, ha='center', va='bottom', fontsize=8)
# labelvalue(rects1)
# labelvalue(rects2)
# ax.text(1.10, 1.001 * threshold, '72.73', ha='center', va='bottom', fontsize=12, color='r')

# 设置图例
# ax.legend((rects1[0], rects2[0]), ('normal pruned', 'graft pruned'), loc=2)
ax.legend(loc=2, frameon=True, fontsize='12')

# plt.savefig('new_accuracy3.jpg', bbox_inches='tight', pad_inches=0.2, dpi=600)
plt.savefig('new_accuracy3-2.jpeg', dpi=600)
plt.show()