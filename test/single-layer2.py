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

# y1 = [92.21, 92.95, 92.51, 92.53, 92.56, 92.56, 92.34, 92.18, 92.31, 91.80]
# y2 = [92.22, 92.64, 92.54, 92.72, 92.85, 92.66, 92.61, 92.65, 92.43, 92.40]

y1 = [93.72, 94.09, 93.91, 93.91, 93.89, 94.05, 94.08, 94.08, 93.83, 93.93]
y2 = [93.73, 94.33, 94.37, 94.09, 94.30, 94.34, 94.40, 94.28, 94.27, 94.31]
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.98, 71.30, 80.48, 87.64, 100]

df['Single-layer filter pruning ratio(%)'] = x
df['Accuracy(%)'] = y1
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='ResNet-56', marker='o')

df['Accuracy(%)'] = y2
ax = sns.lineplot(data=df, x='Single-layer filter pruning ratio(%)', y='Accuracy(%)', label='Adapter-ResNet-56', marker='o')
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ax.set_ylim(91, 93.2)
ax.set_ylim(93, 94.5)
ax.set_xlim(-0.05, 0.95)
ax.set_title('Layer54')

plt.legend(loc=3, fontsize='12')
# plt.savefig('resnet-20-single-layer1.jpg')
plt.savefig('resnet-56-single-layer2.jpg')
plt.show()

