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

# r20_y =     [92.21, 92.70, 92.28, 91.81, 91.47, 91.10, 89.82, 88.88, 87.99, 84.04]
r20_y =     [92.21, 92.47, 92.20, 91.74, 91.15, 90.71, 89.63, 88.75, 86.91, 83.85]
ad15r20_y = [92.22, 92.71, 92.56, 91.98, 91.47, 91.05, 90.59, 89.38, 87.78, 86.18]
print(np.array(ad15r20_y) - np.array(r20_y))

r20_px =   [0, 12.58, 21.51, 31.78, 43.39, 51.84, 62.13, 72.01, 81.92, 90.11]
adr20_px = [0, 12.43, 21.50, 31.50, 43.32, 51.90, 61.97, 72.00, 81.64, 89.95]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.98, 71.30, 80.48, 87.64, 100]

df['Params pruning ratio(%)'] = r20_px
df['Accuracy(%)'] = r20_y
ax = sns.lineplot(data=df, x='Params pruning ratio(%)', y='Accuracy(%)', label='ResNet-20', marker='o')

df['Params pruned ratio(%)'] = adr20_px
df['Accuracy(%)'] = ad15r20_y
# ax = sns.lineplot(data=df, x='Params pruning ratio(%)', y='Accuracy(%)', label='Adapter-ResNet-20-V1', marker='o')
ax = sns.lineplot(data=df, x='Params pruning ratio(%)', y='Accuracy(%)', label='CES-ResNet-20-V1', marker='o')
# ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_ylim(83, 93)
# ax.set_xlim(0, 100)
plt.legend(loc="best")
# plt.savefig('polyline_plot2.jpg')
plt.savefig('CES-polyline_plot2.jpg')
plt.show()

