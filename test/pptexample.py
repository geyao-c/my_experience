import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
sns.set_theme(style='darkgrid')  # 图形主题
# sns.set_theme(style='whitegrid')
df = pd.DataFrame()

y1 = [95, 93, 88, 81, 73, 64, 54, 43, 31, 19]
y2 = [95, 94, 92.5, 90.5, 88, 85.5, 82.5, 79, 75, 70.5]

x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.98, 71.30, 80.48, 87.64, 100]

df['Pruning ratio'] = x
df['Accuracy(%)'] = y1
ax = sns.lineplot(data=df, x='Pruning ratio', y='Accuracy(%)', marker='o', label='line1')

df['Accuracy(%)'] = y2
ax = sns.lineplot(data=df, x='Pruning ratio', y='Accuracy(%)', marker='o', label='line2')

ax.set_ylim(-5, 100)
# # ax.set_ylim(93, 94.5)
ax.set_xlim(-0.05, 0.95)
# ax.set_title('Layer52')

# plt.legend(loc=3, fontsize='12')
plt.savefig('pptexample'+ '.png', dpi=600, pad_inches=0.3, bbox_inches="tight")
# plt.savefig('resnet-56-single-layer2.jpg')
plt.show()

