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

y1 = [93.75, 93.78, 93.76, 93.72, 93.72, 93.70, 93.68, 93.67, 93.66, 93.63]
y2 = [93.73, 93.77, 93.76, 93.76, 93.76, 93.75, 93.74, 93.72, 93.69, 93.64]

x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.98, 71.30, 80.48, 87.64, 100]

df['Single-layer filter pruning ratio(%)'] = x
df['Accuracy(%)'] = y1
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='ResNet-56', marker='o')

df['Accuracy(%)'] = y2
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='Adapter-ResNet-56', marker='o')
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_ylim(92, 94.1)
# ax.set_ylim(93, 94.5)
ax.set_xlim(-0.05, 0.95)
ax.set_title('Layer52')

plt.legend(loc=3, fontsize='12')
plt.savefig('resnet-56-single-layer52.jpg')
# plt.savefig('resnet-56-single-layer2.jpg')
plt.show()

