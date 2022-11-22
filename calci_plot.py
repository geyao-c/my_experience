import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    # dirpath1 = './calculated_ci/93.73_adapter15resnet_56_cifar10'
    # dirpath1 = './calculated_ci/93.99_adapter_vgg_16_bn_cifar10'
    dirpath1 = 'calculated_ci/68.72_adapter19resnet_20_cifar100'
    # dirpath1 = 'calculated_ci/92.22_adapter15resnet_20_cifar10'
    # dirpath1 = './calculated_ci/93.91_adapter_vgg_16_bn_v4_cifar10'
    # dirpath2 = 'calculated_ci/92.21_resnet_20_cifar10'
    # dirpath2 = './calculated_ci/93.59_resnet_56_cifar10'
    # dirpath2 = './calculated_ci/93.96_vgg_16_bn_cifar10'
    dirpath2 = 'calculated_ci/68.70_resnet_20_cifar100'

    file = 'ci_conv10.npy'
    # file = 'ci_conv52.npy'
    filepath1 = os.path.join(dirpath1, file)
    filepath2 = os.path.join(dirpath2, file)

    data1 = np.load(filepath1); data2 = np.load(filepath2)
    data1.sort(); data2.sort()
    data1_sum = np.sum(data1); data2_sum = np.sum(data2)

    delta = 0.1; start = 0.1; end = 1.1
    y1, y2 = [], []
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    while abs(start - end) >= 0.1:
        data1_ratio = np.sum([data1[i] for i in range(0, int(start * len(data1)))])
        data2_ratio = np.sum([data2[i] for i in range(0, int(start * len(data2)))])
        data1_ratio = round(data1_ratio / data1_sum, 2)
        data2_ratio = round(data2_ratio / data2_sum, 2)
        # print('start: ', round(start, 2), data1_ratio, data2_ratio)
        y1.append(data1_ratio); y2.append(data2_ratio)
        start += delta

    # y2[9] = 1.0; y2[8] = 0.85
    print(y1)
    print(y2)
    plt.rcParams['savefig.dpi'] = 600; plt.rcParams['figure.dpi'] = 600
    sns.set_theme(style='darkgrid')  # 图形主题
    df = pd.DataFrame()

    df['Filter pruning ratio'] = x; df['PI'] = y1
    # ax = sns.lineplot(data=df, x='Filter pruning ratio', y='PI', label='Adapter-ResNet-20-V1', marker='o')
    ax = sns.lineplot(data=df, x='Filter pruning ratio', y='PI', label='Adapter-ResNet-20-V2', marker='o')
    # ax = sns.lineplot(data=df, x='Filter pruning ratio', y='PI', label='Adapter-ResNet-56', marker='o')

    df['Filter pruning ratio'] = x; df['PI'] = y2
    ax = sns.lineplot(data=df, x='Filter pruning ratio', y='PI', label='ResNet-20', marker='o')
    # ax = sns.lineplot(data=df, x='Filter pruning ratio', y='PI', label='ResNet-56', marker='o')

    ax.set_title('Layer12')
    # ax.set_title('Layer18')
    # ax.set_title('Layer52')
    plt.legend(loc="best", fontsize=12)
    plt.savefig('calci_plot5.jpg')
    plt.show()

