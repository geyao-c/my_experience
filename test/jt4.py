import os
import numpy as np
import matplotlib.pyplot as plot
from sklearn.manifold import TSNE
from data import cifar10
from models.adapter_resnet_new_three import adapter16resnet_56, adapter17resnet_56
import torch
from thop import profile

def fun1():
    dirpath = '../calculated_ci/94.62_adapter13resnet_56_cifar10'
    dirpath2 = '../calculated_ci/94.54_resnet_56_cifar10'

    filelist = os.listdir(dirpath)
    filelist.sort(key=lambda x: int(x.split('.')[0].split('v')[1]))
    filelist2 = os.listdir(dirpath)
    filelist2.sort(key=lambda x: int(x.split('.')[0].split('v')[1]))
    print(filelist)
    for idx in range(len(filelist)):
        print('======================================')
        # 数值越大表示越重要
        filename = os.path.join(dirpath, filelist[idx])
        dt1 = np.load(filename)

        x = [i for i in range(1, len(dt1) + 1)]
        dt1.sort()
        print(dt1)

        filename2 = os.path.join(dirpath2, filelist2[idx])
        dt2 = np.load(filename2)
        dt2.sort()
        print(dt2)

        if len(x) != len(dt2): continue
        plot.plot(x, dt2, 'g', label='original')
        plot.plot(x, dt1, 'r', label='new')
        plot.title('layer: {}'.format(idx))
        plot.legend()
        plot.savefig('../image/{}.jpg'.format(idx))
        plot.clf()
        # plot.show()

def fun2():
    X = np.array([[0., 0., 0.], [0., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype=np.float64)
    print(type(X))
    print(X.dtype)
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X)
    X_embedded = TSNE(n_components=2, init='pca', random_state=33).fit_transform(X)
    print(X_embedded.shape)
    print(X_embedded)

def fun3():
    x = np.array([])
    y = np.array([4, 5, 6])
    x = np.stack((x, y), axis=0)
    print(x)

def fun4():
    x = np.array([1, 2, 3])
    y = np.array([4, 6])
    x = np.hstack((x, y))
    print(x)

def fun5():
    pass

def fun6():
    model = adapter16resnet_56([0.]*100, 100, [0]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun7():
    model = adapter17resnet_56([0.]*100, 100, [0]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

if __name__ == '__main__':
    # fun1()
    # fun2()
    # fun3()
    # fun4()
    # fun6()
    fun7()