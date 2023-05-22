import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pdframe1 = {
    '42.8% Params reduction': ['Adapter-ResNet-56', 'Adapter-ResNet-56', 'Adapter-ResNet-56',
              'ResNet-56', 'ResNet-56', 'ResNet-56'],
    'Accuracy': [72.73, 71.88, 72.29, 72.60, 71.10, 70.71],
    'Type': ['baseline', 'normal pruned', 'pruned transfer',
             'baseline', 'normal pruned', 'pruned transfer']
}

pdframe2 = {
    '71.3% Params reduction': ['Adapter-ResNet-56', 'Adapter-ResNet-56', 'Adapter-ResNet-56',
              'ResNet-56', 'ResNet-56', 'ResNet-56'],
    'Accuracy': [72.73, 70.26, 70.28, 72.60, 68.59, 68.04],
    'Type': ['baseline', 'normal pruned', 'pruned transfer',
             'baseline', 'normal pruned', 'pruned transfer']
}

df1 = pd.DataFrame(pdframe1)
df2 = pd.DataFrame(pdframe2)

plt.figure(figsize=(12, 7))
sns.set_theme(style="darkgrid")
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600

plt.subplot(1, 2, 1)
plt.ylim(65, 75)
image1 = sns.barplot(x="42.8% Params reduction", y="Accuracy", hue="Type", data=df1)
image1.legend().set_title('')

plt.subplot(1, 2, 2)
plt.ylim(65, 75)
image2 = sns.barplot(x="71.3% Params reduction", y="Accuracy", hue="Type", data=df2)
image2.legend().set_title('')

plt.savefig('./saved_images/image1.jpg', pad_inches=0.3, bbox_inches="tight")
plt.show()
