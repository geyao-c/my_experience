import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os

category = ['干净的', '弄脏的', '擦除后的']
original = [[0., 0., 0., 0.95, 0., 0.05, 0., 0., 0., 0., ],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0.02, 0.01, 0., 0., 0., 0., 0., 0., 0.96, 0.01],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0.01, 0., 0., 0.99, 0., 0., 0.]]

middle = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]

later = [[0., 0., 0., 0.99, 0., 0.01, 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         [0.01, 0.03, 0., 0., 0., 0., 0., 0., 0.96, 0.01],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0.09, 0., 0.02, 0.89, 0., 0., 0.]]

# original = [[0., 0., 0., 0.99, 0., 0.01, 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#             [0., 0., 0., 0.07, 0., 0., 0.93, 0., 0., 0.]]
#
# middle = [[0.39, 0., 0., 0.6, 0., 0.01, 0., 0., 0., 0.],
#           [0.96, 0., 0., 0., 0., 0., 0., 0., 0.04, 0.],
#           [0.99, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#           [0.96, 0.,0., 0.01, 0., 0., 0.03, 0., 0., 0.]]
#
# later = [[0.05, 0., 0., 0.94, 0., 0.01, 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#          [0.89, 0.02, 0., 0., 0., 0., 0., 0., 0.04, 0.04],
#          [0.03, 0., 0.01, 0., 0., 0., 0.97, 0., 0., 0.],
#          [0.14, 0., 0., 0.11, 0.01, 0.03, 0.7, 0., 0., 0.]]
sns.set_theme(style="darkgrid")
df = pd.DataFrame()
df['Classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i, t in enumerate(labels)]
cp = sns.color_palette()
print(len(cp))
for cate in category:
    datas = None
    if cate == '干净的': datas = original
    elif cate == '弄脏的': datas = middle
    elif cate == '擦除后的': datas = later
    for i in range(0, len(datas)):
        df['Accuracy'] =  datas[i]
        ax = sns.barplot(x="Classes", y="Accuracy", data=df, label=labels)

        plt.legend(handles=patches, loc="best")
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600
        dir = os.path.join('图片1', cate)
        if not os.path.exists(dir):
            os.makedirs(dir)
        imagepath = os.path.join(dir, 'sample{}.jpg'.format(i + 1))
        plt.savefig(imagepath)
        plt.show()
        plt.clf()
        break
    break