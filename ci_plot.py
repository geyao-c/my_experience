import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dirpath1 = './calculated_ci/72.19_adapter15resnet_56_cifar100'
    # dirpath2 = './calculated_ci/72.23_resnet_56_cifar10'
    dirpath1 = './calculated_ci/93.46_adapter15resnet_56_cifar10'
    dirpath2 = './calculated_ci/93.59_resnet_56_cifar10'
    filelist1 = os.listdir(dirpath1)
    print(filelist1)
    filelist2 = os.listdir(dirpath2)
    print(filelist2)
    filelist1.sort(key=lambda x: int(x.split('.')[0].split('conv')[1]))
    print(filelist1)
    ci1 = []
    ci2 = []
    for (i, file) in enumerate(filelist1):
        if file == 'ci_conv54.npy' or file == 'ci_conv55.npy':
            continue
        filepath1 = os.path.join(dirpath1, file)
        filepath2 = os.path.join(dirpath2, file)
        data1 = np.load(filepath1)
        data2 = np.load(filepath2)
        data1.sort()
        data2.sort()
        x = [i for i in range(len(data1))]
        # ci1.extend(data1)
        # ci2.extend(data2)
        plt.plot(x, data1, label="ci1")
        plt.plot(x, data2, label="ci2")
        plt.legend()
        plt.title('layer{}'.format(i))
        plt.show()

    # x = [i for i in range(len(ci1))]
    # # 绘制图形
    # plt.plot(x, ci1, label="ci1")
    # plt.plot(x, ci2, label="ci2")
    # print(x)
    # plt.show()
