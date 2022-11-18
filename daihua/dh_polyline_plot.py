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
Accu = [85.75, 88.06, 89.16, 89.71, 90.08, 90.24, 90.42, 90.70]
Rob = [63.87, 63.87, 63.67, 63.38, 62.89, 61.33, 60.74, 60.16]
x = [i * 0.5 for i in range(0, 8)]

df['Params pruned ratio(%)'] = x
df['Accuracy(%)'] = Accu
ax = sns.lineplot(data=df, x='Params pruned ratio(%)', y='Accuracy(%)', label='Accuracy', marker='o')

df['Accuracy(%)'] = Rob
ax = sns.lineplot(data=df, x='Params pruned ratio(%)', y='Accuracy(%)', label='Robustness', marker='o')
# ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_ylabel('Percentage(%)')
ax.set_xlabel('γ')
ax.set_ylim(55, 97)
# ax.set_xlim(0, 100)
plt.legend(loc=2)
plt.savefig('polyline_plot1.jpg')
plt.show()

