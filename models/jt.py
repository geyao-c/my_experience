import torch

from resnet_cifar import resnet_56
from torchsummary import summary
from models.adapter_resnet_new_three import adapter23resnet_56, adapter22resnet_56, \
    adapter24resnet_56
from models.supcon_adapter_resnet import supcon_adapter15resnet_56
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
import torchvision
import utils_append
import numpy as np
import torch.nn.functional as F

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

def fun6():
    model = supcon_adapter15resnet_56([0.]*100, 10, [0.]*100)
    print(model)

def fun7():
    x = np.array([1, 2, 3, 4, 5, 6])
    x = torch.from_numpy(x)
    print(x)

def fun8():
    x = torch.arange(12)
    print(x)
    x = x.reshape(2, 2, 3)
    print(x)
    x1 = x.view(x.shape[0], -1)
    print(x1)
    x2 = torch.flatten(x, 1)
    print(x2)

def fun9():
    model = sl_mlp_adapter15resnet_56([0.]*100, 10, [0.]*100)
    print(model)

def fun10():

    x = torch.tensor([[1, 2, 3], [3, 5, 6]], dtype=torch.float)
    softx = F.softmax(x, dim=1)
    print('softx: ', softx)
    print(np.log(softx))
    print(F.softmax(x))
    log_softx = F.log_softmax(x)
    # print(F.log_softmax(x))
    print('log_softx: ', log_softx)
    print(log_softx.data.exp())
    # print(F.log_softmax(x, dim=1))


if __name__ == '__main__':
    # fun1()
    # fun3()
    # fun4()
    # fun5()
    # fun6()
    # fun7()
    # fun8()
    # fun9()
    fun10()