import torch

from resnet_cifar import resnet_56, resnet_32
from vgg_cifar10 import vgg_16_bn
from adapter_vgg_cifar10 import adapter_vgg_16_bn
from torchsummary import summary
from models.adapter_resnet_new_three import adapter23resnet_56, adapter22resnet_56, \
    adapter24resnet_56, adapter16resnet_32, adapter15resnet_56, adapter15resnet_32
from models.supcon_adapter_resnet import supcon_adapter15resnet_56
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
from models.supcon_adapter_resnet import selfsupcon_adapter15resnet_56
from models.selfsupcon_supcon_adapter_resnet import selfsupcon_supcon_resnet_56
import torchvision
import utils_append
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def fun11():
    model = selfsupcon_adapter15resnet_56([0.]*100, 10, [0.]*100)
    print(model)

def fun12():
    model = selfsupcon_supcon_resnet_56([0.]*100, 10)
    print(model)

def fun13():
    model = resnet_32([0.]*100, 10)
    print(model)

def fun14():
    model = adapter16resnet_32([0.]*100, 10, [0.]*100)
    print(model)

def fun15():
    model = adapter15resnet_56([0.] * 100, 10, [0.] * 100)
    print(model)

def fun16():
    model = adapter15resnet_32([0.] * 100, 10, [0.] * 100)
    print(model)

def fun17():
    original_model = resnet_56([0] * 100, 10)
    str1 = "[0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9"
    sparsity1 = utils_append.analysis_sparsity(str1)
    model1 = resnet_56(sparsity1, 100)

    str2 = "[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9"
    sparsity2 = utils_append.analysis_sparsity(str2)
    model2 = resnet_56(sparsity2, 100)

    # flops, params, flops_ratio, params_ratio = utils_append.cal_params(model1, device, original_model)
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model2, device, original_model)
    print(flops, params, flops_ratio, params_ratio)

def fun18():
    original_model = adapter15resnet_56([0] * 100, 10, [0.] * 100)
    str1 = "[0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9"
    adastr1 = "[0.4]"
    sparsity1 = utils_append.analysis_sparsity(str1)
    adasparsity1 = utils_append.analysis_sparsity(adastr1)
    model1 = adapter15resnet_56(sparsity1, 10, adasparsity1)

    str2 = "[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9"
    adastr2 = "[0.7]"
    sparsity2 = utils_append.analysis_sparsity(str2)
    adasparsity2 = utils_append.analysis_sparsity(adastr2)
    model2 = adapter15resnet_56(sparsity2, 10, adasparsity2)

    # flops, params, flops_ratio, params_ratio = utils_append.cal_params(model1, device, original_model)
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model2, device, original_model)
    print(flops, params, flops_ratio, params_ratio)

def fun19():
    original_model = vgg_16_bn([0.] * 100, 10)
    # str1 = "[0.30]*7+[0.75]*5"
    str1 = "[0.45]*7+[0.78]*5"
    sparsity1 = utils_append.analysis_sparsity(str1)
    model1 = vgg_16_bn(sparsity1, 10)
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model1, device, original_model)
    print(flops, params, flops_ratio, params_ratio)

def fun20():
    original_model = adapter_vgg_16_bn([0.] * 100)
    str1 = "[0.30]*7+[0.75]*5"
    # str1 = "[0.45]*7+[0.78]*5"
    sparsity1 = utils_append.analysis_sparsity(str1)
    model1 = adapter_vgg_16_bn(sparsity1)
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model1, device, original_model)
    print(flops, params, flops_ratio, params_ratio)

if __name__ == '__main__':
    # fun1()
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
    # fun17()
    # fun18()
    # fun19()
    fun20()