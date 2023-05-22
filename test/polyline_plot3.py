import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker


plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
sns.set_theme(style='darkgrid')  # 图形主题
df = pd.DataFrame()

# r20_y =     [92.21, 92.70, 92.28, 91.81, 91.47, 91.10, 89.82, 88.88, 87.99, 84.04]
r20_y =     [68.70, 67.91, 67.40, 66.70, 65.77, 65.07, 63.63, 61.16, 57.15, 50.41]
ad15r20_y = [68.72, 67.67, 67.53, 67.07, 66.24, 65.38, 63.62, 62.11, 58.34, 50.82]
print(np.array(ad15r20_y) - np.array(r20_y))

r20_avg_y =     [68.70, 67.82, 67.02, 66.64, 65.62, 64.64, 63.18, 60.88, 56.90, 49.78]
ad15r20_avg_y = [68.72, 67.38, 67.36, 66.88, 65.83, 65.15, 63.37, 61.73, 57.91, 50.62]
print(np.array(ad15r20_avg_y) - np.array(r20_avg_y))

r20_yy =   [68.70, 67.91, 67.02, 66.70, 65.77, 64.64, 62.96, 61.16, 57.15, 49.78]
adr20_yy = [68.72, 67.67, 67.36, 67.07, 66.24, 65.15, 63.37, 62.11, 58.34, 50.62]
print(np.array(adr20_yy) - np.array(r20_yy))

r20_px =   [0, 12.31, 21.06, 31.11, 42.47, 50.74, 60.81, 70.48, 80.18, 88.20]
adr20_px = [0, 12.13, 20.90, 31.00, 42.39, 50.67, 60.76, 70.32, 80.07, 88.13]
# r20_fx = [0, 13.23, 21.73, 31.68, 47.43, 53.25, 60.81, 71.30, 80.48, 87.64]

df['Params pruning ratio(%)'] = r20_px
df['Accuracy(%)'] = r20_yy
ax = sns.lineplot(data=df, x='Params pruning ratio(%)', y='Accuracy(%)', label='ResNet-20', marker='o')

df['Params pruned ratio(%)'] = adr20_px
df['Accuracy(%)'] = adr20_yy
ax = sns.lineplot(data=df, x='Params pruning ratio(%)', y='Accuracy(%)', label='Adapter-ResNet-20-V2', marker='o')
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ax.set_ylim(49, 70)
# ax.set_xlim(0, 100)
plt.legend(loc="best")
plt.savefig('polyline_plot3.jpg')
plt.show()

