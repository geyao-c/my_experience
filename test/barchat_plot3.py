import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

fake_x = np.array([0, 0.6])
# 柱状图宽度
bar_width = 0.1
baseline = [72.73, 72.60]
pruned_normal = [71.88, 71.10]
pruned_graft = [72.29, 70.71]
x_labels = ['Adapter-ResNet-56', 'ResNet-56']

# 设置画布风格
# plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-white')
fig, ax = plt.subplots(dpi=600)

rects0 = ax.bar(fake_x, baseline, width=bar_width, color='coral', label='baseline')
rects1 = ax.bar(fake_x + bar_width, pruned_normal, width=bar_width, color='green', label='normal pruned')
rects2 = ax.bar(fake_x + 2 * bar_width, pruned_graft, bar_width, color='orange', label='pruned transfer')

# 设置纵坐标间隔
y_major_locator = MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

# 绘制表格横线
# ax.set_axisbelow(True)
ax.grid(color='gray', axis='y', linewidth=1)
# 设置横坐标
ax.set_xticks(fake_x + bar_width)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Accuracy(%)", fontsize=12)
ax.set_xlabel('42.8% Parameter pruned ratio(%)', fontsize=12)

# 隐藏坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 设置柱形图x轴和y轴长度
ax.set_ylim(65, 75)
ax.set_xlim(-0.2, 1.05)
ax.legend(loc=2, frameon=True)
plt.savefig('new_accuracy1.jpeg', dpi=600)
plt.show()