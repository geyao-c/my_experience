import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
sns.set_theme(style='darkgrid')  # 图形主题
df = pd.DataFrame()

names = ['0.25', '1', '4', '8', '12']
x = range(len(names))
# y = [79.94, 80.36, 80.89, 81.45, 81.56]
y = [79.94, 80.20, 80.89, 81.45, 81.56]

df['x'] = x
df['Accuracy(%)'] = y
ax = sns.lineplot(data=df, x='x', y='Accuracy(%)', label='Compression transferability performance', marker='o', color='#DE8452')
# ax.xticks(x, names)
# ax.set_xlabel('k', fontsize=10)
# plt.plot(x, y, 'o-', color='#DF8344')

plt.axhline(y = 80.05, color = 'r', linestyle = 'dashed', label='Baseline')

plt.xticks(x, names)
plt.xlabel('k')
plt.ylabel('Accuracy(%)')
ax = plt.gca()
y_major_locator = MultipleLocator(0.5)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(loc="best")
# plt.savefig('differentk.jpg', pad_inches=0.0, bbox_inches="tight")
plt.savefig('differentk.jpg')
plt.show()

