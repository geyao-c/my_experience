import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
sns.set_theme(style='darkgrid')  # 图形主题
df = pd.DataFrame()

# y1 = [92.21, 92.77, 92.51, 92.59, 92.50, 92.50, 92.44, 92.33, 92.30, 92.23]
# y2 = [92.22, 92.55, 92.42, 92.62, 92.40, 92.38, 92.49, 92.11, 92.20, 92.12]

sy1 = [92.21, 92.77, 92.51, 92.50, 92.49, 92.50, 92.44, 92.33, 92.30, 92.23]
sy2 = [92.22, 92.55, 92.50, 92.61, 92.40, 92.38, 92.49, 92.30, 92.31, 92.28]


x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.98, 71.30, 80.48, 87.64, 100]

df['Single-layer filter pruning ratio(%)'] = x
df['Accuracy(%)'] = sy1
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='ResNet-20', marker='o')

df['Accuracy(%)'] = sy2
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='Adapter-ResNet-20-V1', marker='o')
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_ylim(90, 93.2)
# ax.set_ylim(93, 94.5)
ax.set_xlim(-0.05, 0.95)
ax.set_title('Layer16')

plt.legend(loc=3, fontsize='12')
plt.savefig('resnet-20-single-layer2.jpg')
# plt.savefig('resnet-56-single-layer2.jpg')
plt.show()

