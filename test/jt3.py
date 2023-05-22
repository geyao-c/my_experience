import numpy as np

from resnet import ResNet, BasicBlock
from thop import profile
import utils_append
import torch
from models.adapter_resnet_new import adapter5resnet_56, adapter6resnet_56
import torch.nn as nn
from models.adapter_resnet_new_new import adapter8resnet_56
from models.resnet_cifar import resnet_80, resnet_56
from torchsummary import summary
from models.adapter_resnet_new_three import adapter9resnet_56, adapter10resnet_56, \
    adapter11resnet_56, adapter12resnet_56, adapter14resnet_56, adapter15resnet_56, \
    adapter18resnet_56, adapter19resnet_56, adapter20resnet_56, adapter21resnet_56, \
    adapter22resnet_56
from models.adapter_resnet_new_three import adapter13resnet_56

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fun1():
    # resnet
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    for i in range(13, 14):
        layers = i * 6 + 2
        model = ResNet(BasicBlock, layers, [0.] * 100, 100)
        flops, params = profile(model, (input_image,))
        print('layers: {}, flops: {}, params: {}'.format(layers, flops, params))

def fun2():
    layers = 80
    repeat = (80 - 2) // 6
    # sparsity = '[0.]+[0.15]*2+[0.4]*{}+[0.4]*{}+[0.4]*{}'.format(repeat, repeat, repeat)
    sparsity = '[0.]+[0.4]*2+[0.5]*{}+[0.6]*{}+[0.75]*{}'.format(repeat, repeat, repeat)
    sparsity = utils_append.analysis_sparsity(sparsity)
    print('sparsity: {}'.format(sparsity))
    model = ResNet(BasicBlock, layers, sparsity, 100)
    original_model = ResNet(BasicBlock, layers, [0.]*100, 100)
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_model)
    print('layers: {}, flops: {}, params: {}, flops ratio: {}, params ratio: {}'.format(
        layers, flops, params, flops_ratio, params_ratio))

def fun3():
    model = adapter5resnet_56([0.]*100, 100, [0.]*100)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    # print(model)
    print('flops: {}, params: {}'.format(flops, params))

def fun4():
    model = adapter5resnet_56([0.] * 100, 100, [0.] * 100)
    params = model.state_dict()
    print(type(params))
    for key in params.keys():
        print(key)

def fun5():
    input_tensor = torch.randn((1, 16, 16, 16))
    model = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
                          nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1))
    output_tensor = model(input_tensor)
    print(model)
    print(output_tensor.shape)

def fun6():
    input_image_size = 16
    input_image = torch.randn(1, 16, input_image_size, input_image_size)
    model = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False),
                          nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False))
    flops, params = profile(model, (input_image, ))
    print("flops1: {}, params1: {}".format(flops, params))

    model2 = nn.Sequential(nn.Conv2d(16, 16 * 9, kernel_size=(1, 1), padding=0, bias=False),
                          nn.Conv2d(16 * 9, 16, kernel_size=(1, 1), padding=0, bias=False))
    flops, params = profile(model2, (input_image, ))
    print("flops2: {}, params2: {}".format(flops, params))

def fun7():
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    model1 = adapter8resnet_56(sparsity=[0.]*100, num_classes=10, adapter_sparsity=[0.]*100)
    summary(model1, (3, 32, 32), device='cpu')
    print(model1)
    flops1, params1 = profile(model1, (input_image, ))
    print("flops1: {}, params1: {}".format(flops1, params1))

    model2 = resnet_80(sparsity=[0.]*100, num_classes=10)
    # print(model2)
    flops2, params2 = profile(model2, (input_image,))
    print("flops2: {}, params1: {}".format(flops2, params2))

def fun8():
    model = adapter8resnet_56(sparsity=[0.]*100, num_classes=10, adapter_sparsity=[0.]*100)
    print(model)
    params = model.state_dict()
    for key in params.keys():
        print(key)

def fun9():
    x = torch.randn((3, 3, 16, 16)).to(device)
    # print(x)
    print(type(x))
    print(x.device)

def fun10():
    x = torch.tensor([1, 2, 3]).to(device)
    print(x)
    print(x.device)

def fun11():
    x = np.load('../calculated_ci/94.96_adapter8resnet_56_cifar10/ci_conv55.npy')
    print(x)
    x2 = np.load('../calculated_ci/94.96_adapter8resnet_56_cifar10_gpu/ci_conv55.npy')
    print(x2)

def fun12():
    """
    = = =
    = = =
    = = +
    = + =
    = = =
    = = +
    = = =
    = + =
    + = +
    """
    pass

def fun13():
    x = range(1, 10)
    print(x)
    for i in range(1, 10):
        print(i)
        i += 1
        print(i)

def fun14():
    model = adapter9resnet_56([0.]*100, 100, [0.]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image, ))
    print('flops1: {}, params1: {}'.format(flops, params))
    # summary(model, (3, 32, 32), device='cpu')

    model2 = resnet_56([0.]*100, 100)
    flops2, params2 = profile(model2, (input_image, ))
    print('flops2: {}, params2: {}'.format(flops2, params2))
    # summary(model2, (3, 32, 32), device='cpu')

def fun15():
    model = adapter10resnet_56([0.]*100, 100, [0.]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun16():
    model = adapter11resnet_56([0.] * 100, 100, [0.] * 100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun17():
    model = adapter12resnet_56([0.] * 100, 100, [0.] * 100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun18(x=None):
    if x is None:
        x = [1, 2, 3]
    print(x)

def fun19():
    model = adapter13resnet_56([0.]*100, 100, [0.]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun20():
    model = adapter14resnet_56([0.] * 100, 10, [0.] * 10)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun21():
    model = adapter15resnet_56([0.] * 100, 100, [0.] * 10)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun22():
    x = 1.23436
    x = np.round(x, 3)
    print(x)

def fun23():
    model = adapter18resnet_56([0.]*100, 10, [0.]*100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun24():
    model = adapter19resnet_56([0.] * 100, 100, [0.] * 100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun25():
    model = adapter20resnet_56([0.] * 100, 10, [0.] * 100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun26():
    model = adapter21resnet_56([0.] * 100, 100, [0.] * 100)
    print(model)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

def fun27():
    model = adapter22resnet_56([0.] * 100, 10, [0.] * 100)
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
    # fun5()
    # fun6()
    # fun7()
    # fun8()
    # fun9()
    # fun10()
    # fun11()
    # fun13()
    # fun14()
    # fun15()
    # fun16()
    # fun17()
    # fun18()
    # fun19()
    # fun20()
    # fun21()
    # fun22()
    # fun23()
    # fun24()
    # fun25()
    # fun26()
    fun27()