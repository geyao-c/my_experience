import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# pd.options.display.notebook_repr_html = False  # 表格显示
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
df = pd.DataFrame()

yaccuracy = [[82.71, 83.18, 83.49, 83.64, 82.53, 81.54, 82.78, 81.69, 82.53],
             [100, 96.64, 95.83, 91.29, 90.2, 88.94, 85.38, 81.99, 84.36],
             [80.77, 80.44, 80.46, 81.29, 80.99, 79.94, 81.17, 80.16, 81.31],
             [2.12, 27.56, 21.47, 27.09, 38.07, 34.03, 38.14, 36.2, 20.89]]
# yaccuracy = [[82.14, 83.54, 83.67, 83.26, 83.16, 82.59, 82.91, 82.16, 81.62],
#              [99.98, 99.84, 99.56, 98.49, 99.26, 98.83, 96.27, 98.21, 97.28],
#              [0.92, 33.86, 1.53, 7.48, 41.44, 10.44, 6.84, 6.02, 21.62],
#              [80.36, 81.6, 81.24, 80.67, 82.22, 82.38, 82.49, 81.44, 81.42]]
yaccuracy = [[82.14, 83.54, 83.67, 83.26, 83.16, 82.59, 82.91, 82.16, 81.62],
             [99.98, 99.84, 99.56, 98.49, 99.26, 98.83, 96.27, 98.21, 97.28],
             [80.36, 81.6, 81.24, 80.67, 82.22, 82.38, 82.49, 81.44, 81.42],
             [0.92, 33.86, 1.53, 7.48, 41.44, 10.44, 6.84, 6.02, 21.62]]
df['Beta'] = [0, 50, 100, 150, 200, 250, 300, 350, 400]
sns.set_theme(style='darkgrid')  # 图形主题

df['Accuracy'] = yaccuracy[0]
ax = sns.lineplot(data=df, x='Beta', y='Accuracy', label='clean accuracy', palette='r')
df['Accuracy'] = yaccuracy[1]
ax = sns.lineplot(data=df, x='Beta', y='Accuracy', label='attack accuracy', palette='r')
df['Accuracy'] = yaccuracy[2]
ax = sns.lineplot(data=df, x='Beta', y='Accuracy', label='clean accuracy(After NAD)', palette='r')
df['Accuracy'] = yaccuracy[3]
ax = sns.lineplot(data=df, x='Beta', y='Accuracy', label='attack accuracy(After NAD)', palette='r')

ax.set_ylim(0, 138)
plt.legend(loc="best")
plt.savefig('image1.jpg')
plt.show()
