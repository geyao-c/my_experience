import torch

from resnet_cifar import resnet_56
from torchsummary import summary
from models.adapter_resnet_new_three import adapter23resnet_56, adapter22resnet_56, \
    adapter24resnet_56
import torchvision
import utils_append

def fun2():
    torchvision.models.resnet50()

def fun1():
    model = resnet_56([0.]*100, 47, dataset='dtd')
    print(model)
    summary(model, (3, 64, 64))
    flops, params, _, _ = utils_append.cal_params(model, torch.device('cpu'), None, 64)
    print('flops: {}, params: {}'.format(flops, params))

def fun3():
    model = adapter22resnet_56([0.] * 100, 47, [0.]*100, dataset='dtd')
    print(model)
    summary(model, (3, 64, 64))
    flops, params, _, _ = utils_append.cal_params(model, torch.device('cpu'), None, 64)
    print('flops: {}, params: {}'.format(flops, params))

def fun4():
    x = dict()
    y = x
    print(id(x))
    print(id(y))

def fun5():
    model = adapter24resnet_56([0.] * 100, 10, [0.] * 100)
    print(model)
    summary(model, (3, 64, 64))
    flops, params, _, _ = utils_append.cal_params(model, torch.device('cpu'), None, 64)
    print('flops: {}, params: {}'.format(flops, params))

if __name__ == '__main__':
    # fun1()
    # fun3()
    # fun4()
    fun5()