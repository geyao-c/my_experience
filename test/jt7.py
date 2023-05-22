import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary
import numpy as np
from models.sl_mlp_resnet_cifar import sl_mlp_resnet_56
from models.adapter_resnet_new_three import adapter17resnet_56

class BaseModel(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = out + x
        # out = self.relu2(out)
        return out

class Model1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Model1, self).__init__()
        self.branch = BaseModel(in_channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.branch(x)
        out = self.relu(out + x)
        return out

class Model2(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Model2, self).__init__()
        self.branch1 = BaseModel(in_channels, mid_channels)
        self.branch2 = BaseModel(in_channels, mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = self.relu1(out1 + out2 + x)
        return out

def fun1():
    input_image_size = 32
    input_image = torch.randn(1, 8, input_image_size, input_image_size)
    model = Model1(8, 8 * 8)
    flops, params = profile(model, (input_image,))
    print('flops1: {}, params1: {}'.format(flops, params))

    model2 = Model2(8, 8 * 4)
    flops, params = profile(model2, (input_image,))
    print('flops2: {}, params2: {}'.format(flops, params))

def fun2():
    model = Model1(8, 8 * 8)
    # model = Model2(8, 8 * 4)
    input_size = (8, 32, 32)
    summary(model, input_size, -1, device='cpu')

def fun3():
    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5, 6]])
    x = np.append(x, y, axis=0)
    print(x)

def fun4():
    x = [3, 4]
    x = np.array(x)
    print(x)
    y = []
    y.append(x.tolist())
    print(y)

def fun5():
    x = [1, 2, 3]
    y = [7]
    print(y.extend(x))
    print(y)

def fun6():
    model = sl_mlp_resnet_56([0.]*100, 10)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    flops, params = profile(model, (input_image,))
    print(model)
    print('flops1: {}, params1: {}'.format(flops, params))

def fun7():
    model = adapter17resnet_56([0.]*100, 10, [0]*100)
    print(model)

if __name__ == '__main__':
    # fun1()
    # fun2()
    # fun3()
    # fun4()
    # fun5()
    # fun6()
    fun7()