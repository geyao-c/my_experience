import torch
from models.resnet_imagenet import resnet_34, resnet_50
from models.adapter_resnet_imagenet import adapter15resnet_34
from torchsummary import summary
from models.resnet_cifar import resnet_56
from models.mobilenetv2 import mobilenet_v2, mobilenet_v2_change, mobilenet_v2_change_2
from models.efficientnet import efficientnet_b0, efficientnet_b0_changed, efficientnet_b0_changed_v2, \
    efficientnet_b0_changed_v3, efficientnet_b0_changed_v4
from models.resnet_cifar import resnet_56
from thop import profile
from models.adapter_efficientnet import adapter_efficientnet_b0_changed_v4

def fun1():
    model = resnet_34([0.0]*100)
    print(model)
    # summary(model, (3, 224, 224))
    print(model.conv1)
    print(type(model.conv1.weight))
    print(model.conv1.weight.shape)
    print(model.conv1.weight.data.shape)
    x = model.conv1.weight.data
    x = x.view(x.size(0), -1)
    print(x.shape)
    sum_of_kernel = torch.sum(torch.abs(x), dim=1)
    print(sum_of_kernel.shape)
    print(sum_of_kernel)

def fun2():
    model = resnet_50([0.]*100)
    # print(model)
    # summary(model, (3, 224, 224))
    # print(model)

def fun3():
    model = adapter15resnet_34([0.]*100, [0.]*100)
    print(model)
    # summary(model, (3, 224, 224))

def fun4():
    model_path = '../pretrained_models/79.468_resnet34_B.pth.tar'
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    print(type(ckpt))
    print(ckpt.keys())
    print(type(ckpt['state_dict']))
    print(ckpt['state_dict'].keys())

def fun5():
    model = resnet_34()
    print(model)
    print(type(len(model.layer1)))
    print(len(model.layer1))

def fun6():
    x = []
    x += [5]
    print(x)

def fun7():
    model = resnet_56([0.]*100, 10)
    summary(model, (3, 32, 32), -1, 'cpu')

def fun8():
    # model = mobilenet_v2([0.]*100, classes=10)
    # summary(model, (3, 32, 32), -1, 'cpu')
    model = mobilenet_v2_change([0.]*100, 10)
    summary(model, (3, 32, 32), -1, 'cpu')
    # model = mobilenet_v2_change_2([0.]*100,10)
    # summary(model, (3, 32, 32), -1, 'cpu')

def fun9():
    # model = efficientnet_b0_changed([0.]*100, 10)
    model = efficientnet_b0([0.]*100, 10)
    # print(model)
    summary(model, (3, 32, 32), -1, 'cpu')
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print(macs, params)

def fun10():
    model = mobilenet_v2([0.]*100, 10)
    print(model)
    summary(model, (3, 224, 224), -1, 'cpu')

def fun11():
    model = resnet_56([0.]*100, 10)
    summary(model, (3, 32, 32), -1, 'cpu')
    # input = torch.randn(1, 3, 32, 32)
    # macs, params = profile(model, inputs=(input,))
    # print(macs, params)

def fun12():
    model = efficientnet_b0_changed_v2([0.]*100, 10)
    summary(model, (3, 32, 32), -1, 'cpu')
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print(macs, params)

def fun13():
    model = efficientnet_b0_changed_v4([0.]*100, 10)
    summary(model, (3, 32, 32), -1, 'cpu')
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print(macs, params)

def fun14():
    model = efficientnet_b0_changed_v2([0.]*100, 10)
    model_path = '../pretrained_models/94.32_efficientnet_b0_v2_cifar10.pth.tar'
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))

def fun15():
    model = adapter_efficientnet_b0_changed_v4([0.] * 100, 10, [0.]*100)
    summary(model, (3, 32, 32), -1, 'cpu')
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print(macs, params)

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
    fun13()
    # fun15()

# layer4.0.conv1.weight
# 126554816.0, 853018
# 8759196,     4020358
# 416018556,   4020358
# 287719548
# 380653692
# efficientnet_b0_changed_v4
# 280051836.0 3918982.0
# adapter_efficientnet_b0_changed_v4
# 277569564.0 3880102.0