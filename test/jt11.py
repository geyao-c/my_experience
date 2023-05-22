import os.path
import jieba
import torch
import utils_append
import argparse
import time
import datetime
import logging
import numpy as np
from models.resnet_cifar import resnet_56, resnet_20
from models.vgg_cifar10 import vgg_16_bn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from models import vgg_cifar10
from models import adapter_vgg_cifar10, adapter_resnet_new_three
from models.adapter_resnet_new_three import adapter15resnet_20


def fun1():
    str = "深度学习是一种计算机自动学习算法，不用像符号主义的方法一样人工定义规则。"
    result = list(jieba.cut(str))
    print(result)

def fenci(str):
    # jieba.cut函数接受三个输入参数，分别是待分词字符串，cut_all形参表示是否采用
    # 全模式，默认为false，HMM形参表示是否使用HMM模型，默认为false
    ans = jieba.cut(str)
    ans = list(ans)
    return ans

def fun2():
    x = torch.tensor(1.2345)
    print(x)
    print(x.item())
    x = round(x.item(), 2)
    print(x)

def fun3():
    parser = argparse.ArgumentParser("Train pruned pipline")
    parser.add_argument('--result_dir', type=str, default='./log/test',
                        help='results path for saving models and loggers')
    args = parser.parse_args()
    mid_result_dir = args.result_dir
    while 1:
        print('----------------------------------------')
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 当前时间
        args.result_dir = os.path.join(mid_result_dir, now)
        logger, writer = utils_append.lgwt_construct(args)
        logger.info('time now is {}'.format(now))
        logger.info('***************************')
        print(logger.handlers)
        logger.handlers = []
        time.sleep(3)

def fun4():
    x = np.array([1, 2, 3])
    print(x.shape)

def fun5():
    model = resnet_56([0.]*100, 10)
    # print(model)
    summary(model, (3, 32, 32))

def fun6():
    model = vgg_16_bn([0.]*100)
    print(model)
    summary(model, (3, 32, 32))

def fun7():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data = [[1, 2, 3, 4, 5, 6],
            [2, 4, 6, 8, 10, 12],
            [1, 3, 5, 7, 9, 11],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [3, 6, 9, 12, 15, 18],
            [4, 8, 12, 16, 20, 24],
            [5, 10, 15, 20, 25, 30],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            [1, 4, 7, 10, 13, 16],
            [2, 5, 8, 11, 14, 17]]
    y = [np.mean(i) for i in data]
    print(y)

    # 绘制图形
    plt.plot(x, y, linewidth=1, color="orange", marker="o", label="Mean value")
    # 算标准差
    yTop = [y[i] + np.std(data[i]) for i in range(len(data))]
    yBottom = [y[i] - np.std(data[i]) for i in range(len(data))]
    plt.fill_between(x, yTop, yBottom, color="lightgreen", label="Standard deviation")  # 填充色块
    # 设置横纵坐标
    plt.xticks([0, 2, 4, 6, 8, 10, 12])
    plt.yticks([0, 5, 10, 15, 20, 25])
    plt.legend(["Mean value", "Standard deviation"], loc="upper left")  # 设置线条标识
    plt.grid()  # 设置网格模式
    # 设置每个点上的数值
    for i in range(10):
        plt.text(x[i], y[i], y[i], fontsize=12, color="black", style="italic", weight="light",
                 verticalalignment='center', horizontalalignment='right', rotation=90)
    plt.show()

def fun8():

    # x = np.array([93.59, 93.93, 93.75, 93.48, 94.01])
    x = np.array([72.77, 72.60, 72.07, 72.03, 72.69])
    x_mean = np.mean(x)
    x_var = np.std(x)
    print(x_mean)
    print(x_var)

    # y = np.array([93.77, 93.46, 93.96, 93.72])
    y = np.array([72.66, 72.73, 72.61, 73.02])
    y_mean = np.mean(y)
    y_var = np.std(y)
    print(y_mean)
    print(y_var)

def fun9():
    x = np.array([2, 4, 6, 8, 10])
    idx_x = np.argsort(-x)
    print(idx_x)

def fun10():
    model = vgg_16_bn([0.]*100)
    print(model)
    summary(model, (3, 32, 32))

def fun11():
    model = adapter_vgg_cifar10.adapter_vgg_16_bn([0.]*100)
    print(model)
    summary(model, (3, 32, 32))

def fun12():
    model = adapter_vgg_cifar10.adapter_vgg_16_bn_v2([0.] * 100)
    print(model)
    summary(model, (3, 32, 32))

def fun13():
    model = adapter_vgg_cifar10.adapter_vgg_16_bn_v3([0.] * 100)
    print(model)
    summary(model, (3, 32, 32))

def fun14():
    model = adapter_vgg_cifar10.adapter_vgg_16_bn_v4([0.] * 100)
    print(model)
    summary(model, (3, 32, 32))

def fun15():
    model = adapter_resnet_new_three.adapter15resnet_20([0.]*100, 10, [0.]*100)
    print(model)

# 计算encoder裁剪比例
def fun16():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparsity = '[0.]+[0.3]*2+[0.4]*3+[0.5]*3+[0.6]*3'
    sparsity = utils_append.analysis_sparsity(sparsity)
    model = resnet_20(sparsity, 10)
    original_model = resnet_20([0.] * 100, 10)

    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_model)
    print(flops, params)
    print(flops_ratio, params_ratio)

def fun17():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparsity = '[0.]+[0.3]*2+[0.4]*3+[0.5]*3+[0.6]*3'
    adapter_sparsity = '0.6'
    sparsity = utils_append.analysis_sparsity(sparsity)
    adapter_sparsity = utils_append.analysis_sparsity(adapter_sparsity)
    model = adapter15resnet_20(sparsity, 10, adapter_sparsity)
    original_model = adapter15resnet_20([0.]*100, 10, [0.]*100)

    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_model)
    print(flops, params)
    print(flops_ratio, params_ratio)

if __name__ == '__main__':
    # fun1()
    # fun2()
    # fun3()
    # fun4()
    # fun5()
    # fun6()
    # fun7()
    # fun8()
    # fun9()
    # fun10()
    # fun11()
    # fun12()
    # fun13()
    # fun14()
    # fun15()
    # fun16()
    fun17()