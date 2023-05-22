import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

category = ['original', 'middle', 'later']
original = [[0., 0., 0., 0.95, 0., 0.05, 0., 0., 0., 0., ],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0.02, 0.01, 0., 0., 0., 0., 0., 0., 0.96, 0.01],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0.01, 0., 0., 0.99, 0., 0., 0.]]

sns.set_theme(style="darkgrid")
df = pd.DataFrame()
df['Classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for cate in category:
    datas = None
    if cate == 'original': datas = original
    else: raise()
    for i in range(0, len(datas)):
        df['Accuracy'] =  [0., 0., 0., 0.95, 0., 0.05, 0., 0., 0., 0.]
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        ax = sns.barplot(x="Classes", y="Accuracy", data=df, label=labels)
        patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(labels)]
        plt.legend(handles=patches, loc="upper right")
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600
        plt.savefig('example.jpg')
        plt.show()