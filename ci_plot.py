import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dirpath1 = './calculated_ci/72.64_adapter15resnet_56_cifar100'
    # dirpath2 = './calculated_ci/72.23_resnet_56_cifar100'
    # dirpath1 = './calculated_ci/72.73_adapter15resnet_56_cifar100'
    # dirpath2 = './calculated_ci/72.60_resnet_56_cifar100'
    dirpath1 = './calculated_ci/93.73_adapter15resnet_56_cifar10'
    # dirpath1 = './calculated_ci/93.99_adapter_vgg_16_bn_cifar10'
    # dirpath1 = './calculated_ci/93.91_adapter_vgg_16_bn_v4_cifar10'
    # dirpath1 = 'calculated_ci/92.22_adapter15resnet_20_cifar10'
    # dirpath1 = 'calculated_ci/68.72_adapter19resnet_20_cifar100'
    # dirpath1 = './calculated_ci/93.33_adapter22resnet_56_cifar10'
    dirpath2 = './calculated_ci/93.59_resnet_56_cifar10'
    # dirpath2 = './calculated_ci/93.96_vgg_16_bn_cifar10'
    # dirpath2 = 'calculated_ci/92.21_resnet_20_cifar10'
    # dirpath2 = 'calculated_ci/68.70_resnet_20_cifar100'

    filelist1 = os.listdir(dirpath1)
    print(filelist1)
    filelist2 = os.listdir(dirpath2)
    print(filelist2)
    filelist1.sort(key=lambda x: int(x.split('.')[0].split('conv')[1]))
    print(filelist1)
    ci1 = []
    ci2 = []
    total_sum1, total_sum2 = 0, 0
    for (i, file) in enumerate(filelist1):
        # if file == 'ci_conv54.npy' or file == 'ci_conv55.npy':
        #     continue
        filepath1 = os.path.join(dirpath1, file)
        filepath2 = os.path.join(dirpath2, file)
        data1 = np.load(filepath1)
        data2 = np.load(filepath2)

        # idx_1 = np.argsort(data1)
        # idx_2 = np.argsort(-data1)
        # print('idx_1: ', idx_1)
        # print('idx_2: ', idx_2)

        data1.sort()
        data2.sort()
        data1_sum = np.sum(data1)
        data2_sum = np.sum(data2)
        total_sum1 += data1_sum
        total_sum2 += data2_sum

        data1_ratio = np.sum([data1[i] for i in range(0, int(0.4 * len(data1)))])
        data2_ratio = np.sum([data2[i] for i in range(0, int(0.4 * len(data2)))])
        # 0.05，0.09，0.14，0.17,0.21，0.23，0.22，0.21，0.13
        print(data1_ratio, data2_ratio)
        try:
            data1_ratio = round(data1_ratio / data1_sum, 2)
            data2_ratio = round(data2_ratio / data2_sum, 2)
            print(i, data1_ratio, data2_ratio, ':', data1_sum, data2_sum)
            # print('{}: ci_1_ratio: {}, ci_2_ratio: {}'.format(i, round(data1_ratio, 2), round(data2_ratio, 2)))
        except:
            pass
        total_sum1 = round(total_sum1, 2)
        total_sum2 = round(total_sum2, 2)

        x1 = [i for i in range(len(data1))]
        x2 = [i for i in range(len(data2))]
        # ci1.extend(data1)
        # ci2.extend(data2)
        plt.plot(x1, data1, label="ci1")
        plt.plot(x2, data2, label="ci2")
        plt.legend()
        plt.title('layer{}'.format(i))
        plt.savefig('./image/{}.jpg'.format(i))
        plt.clf()
        plt.show()
    print('total sum1: {}, total sum2: {}'.format(total_sum1, total_sum2))

    # x = [i for i in range(len(ci1))]
    # # 绘制图形
    # plt.plot(x, ci1, label="ci1")
    # plt.plot(x, ci2, label="ci2")
    # print(x)
    # 1、前面52层
    # 2、所有的层的，
    # 3、
    # plt.show()

