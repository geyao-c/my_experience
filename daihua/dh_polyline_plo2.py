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
Accu = [88.67, 88.67, 88.67, 88.67, 88.67, 88.67, 88.67, 88.67]
Rob = [61.83, 70.16, 71.78, 72.30, 72.42, 72.51, 72.53, 72.53]
x = [i * 0.1 for i in range(0, 8)]

df['x'] = x
df['y'] = Accu
ax = sns.lineplot(data=df, x='x', y='y', label='Accuracy', marker='o')

df['y'] = Rob
ax = sns.lineplot(data=df, x='x', y='y', label='Robustness', marker='o')
# ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_ylabel('Percentage(%)')
ax.set_xlabel('δ')
ax.set_ylim(55, 97)
# ax.set_xlim(0, 100)
plt.legend(loc=2)
plt.savefig('polyline_plot2.jpg')
plt.show()

