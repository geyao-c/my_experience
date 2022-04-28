import torch

from resnet_cifar import resnet_56
from torchsummary import summary
import torchvision
import utils_append

def fun2():
    torchvision.models.resnet50()

def fun1():
    model = resnet_56([0.]*100, 10, dataset='dtd')
    print(model)
    summary(model, (3, 64, 64))
    flops, params, _, _ = utils_append.cal_params(model, torch.device('cpu'), None, 64)
    print('flops: {}, params: {}'.format(flops, params))

if __name__ == '__main__':
    fun1()