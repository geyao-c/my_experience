import argparse
from data import tinyimagenet, mnist
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from models import resnet_cifar
import utils_append
from thop import profile
import torch
from torchsummary import summary
from models.resnet_cifar import resnet_56
from models import resnet_tinyimagenet
from models.selfsupcon_supcon_adapter_vgg import selfsupcon_supcon_adapter_vgg_16_bn
from models.selfsupcon_supcon_adapter_resnet import selfsupcon_supcon_adapter15resnet_20
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets
from models.adapter_resnet_imagenet import adapter15resnet_34
from models.resnet_imagenet import resnet_50

def fun1():
    sparsity = '[0.0]*9+[0.2]*9+[0.2]*9'
    sparsity = utils_append.analysis_sparsity(sparsity)
    print(sparsity)

def fun2():
    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--batch_size', type=int, default=32, help='path to dataset')
    parser.add_argument('--data_dir', type=str, default='/Users/chenjie/dataset/tiny-imagenet-200', help='path to dataset')
    parser.add_argument('--gpu', type=str, default=None, help='path to dataset')

    args = parser.parse_args()
    dataset = tinyimagenet.Data(args)
    train_loader = dataset.train_loader
    print(len(train_loader))
    print(200 * 500)

def fun3():
    dirpath = '/Users/chenjie/dataset/tiny-imagenet-200/train/n04417672'
    filenames = os.listdir(dirpath)
    print(filenames)

    scale_size = 32
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(32),
        transforms.CenterCrop(32),
        # transforms.Resize(32),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(scale_size),
        # transforms.ToTensor(),
        # normalize,
    ])

    for idx, filename in enumerate(filenames):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)
        image.show()
        transformed_image = transform(image)
        transformed_image.show()
        if idx  >= 0: break
        # break

def fun4():
    # sparsity = '[0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9'
    # sparsity = '[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9'
    sparsity = '[0.]*100'
    sparsity = utils_append.analysis_sparsity(sparsity)
    model = resnet_cifar.resnet_56(sparsity, 10)
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size)
    total_ops, total_params = profile(model, inputs=(input_image, ))
    print('total ops: {}, total params: {}'.format(total_ops, total_params))

def fun5():
    dirpath = '/Users/chenjie/dataset/tiny-imagenet-200/train/n04417672'
    filenames = os.listdir(dirpath)
    transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8)
    ])
    for idx, filename in enumerate(filenames):
        filepath = os.path.join(dirpath, filename)
        image = Image.open(filepath)
        image.show()
        transformed_image = transform(image)
        transformed_image.show()
        if idx  >= 0: break
        # break

def fun6():
    # sparsity = '[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9'
    sparsity = '[0.]*100'
    sparsity = utils_append.analysis_sparsity(sparsity)
    model = resnet_56(sparsity, 200)
    # model = resnet_tinyimagenet.resnet_tinyimagenet_56(sparsity, 200)
    # summary(model, (3, 64, 64))
    summary(model, (3, 32, 32))

def fun7():
    model = selfsupcon_supcon_adapter_vgg_16_bn([0.]*100)
    summary(model, (3, 32, 32))
    # print(model)

def fun8():
    parser = argparse.ArgumentParser("MNIST loader")
    parser.add_argument('--batch_size', type=int, default=32, help='path to dataset')
    parser.add_argument('--data_dir', type=str, default='../data')
    args = parser.parse_args()

    train_loader, val_loader = mnist.load_cifar_data(args=args)
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        break

def fun9():
    x = np.array([1, 2])
    print(x)
    print(x + 1)

def fun10():
    model = selfsupcon_supcon_adapter15resnet_20([0.] * 100, 10, [0.] * 100)
    print(model)

def fun11():
    sparsity = '[0.]*100'
    sparsity = utils_append.analysis_sparsity(sparsity)
    weight = torch.load('../pretrained_models/94.78_resnet_56_cifar10.pth.tar', map_location='cpu')
    print(type(weight))
    print(weight.keys())

    model = resnet_cifar.resnet_56(sparsity, 10)
    model.load_state_dict(weight['state_dict'])
    print(type(model.conv1))
    print(model.conv1.weight)

def cal_l1(convl):
    x = torch.sum(torch.abs(convl.view(convl.size(0), -1)), dim=1)
    return x

def fun12():
    x = np.array([[[0.5, 0.1, 0.4], [-0.1, 0.6, 0.1], [0.2, -0.6, 0.5]],
                  [[0.7, 0.1, 0.8], [0.6, -0.7, 0.0], [0.5, 0.9, -0.2]]])
    x = torch.Tensor(x)
    print(x)
    print(x.shape)
    x = cal_l1(x)
    print(x)

def fun13():
    x = np.array([[[0.5]], [[0.1]], [[0.4]], [[-0.1]], [[0.6]], [[0.1]], [[0.2]], [[-0.6]], [[0.5]],
                  [[0.7]], [[0.1]], [[0.8]], [[0.6]], [[-0.7]], [[0.0]], [[0.5]], [[0.9]], [[-0.2]]
                  ])
    x = torch.Tensor(x)
    print(x.shape)
    x = cal_l1(x)
    print(x)

def reduced_1_row_norm(input, row_index, data_index):
    # input shape is (H, C, H * W)
    # 把这一行赋值为0
    input[data_index, row_index, :] = torch.zeros(input.shape[-1])
    # 求矩阵的核范数
    m = torch.norm(input[data_index, :, :], p = 'nuc').item()
    return m

def ci_score(path_conv):
    # 保留4位小数
    # conv_output = torch.tensor(np.round(np.load(path_conv), 4))
    conv_output = path_conv
    # print(conv_output)
    print(conv_output.shape)
    # 参数的意义分别为: batch, channel, height, width
    # conv_reshape shape is (N, C, H * W)
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)

    r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
    for i in range(conv_reshape.shape[0]):
        for j in range(conv_reshape.shape[1]):
            r1_norm[i, j] = reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)

    ci = np.zeros_like(r1_norm)

    for i in range(r1_norm.shape[0]):
        original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        ci[i] = original_norm - r1_norm[i]

    # return shape: [batch_size, filter_number]
    return ci

def fun14():
    conv1 = np.array([[[[0.5, 0.1, 0.4], [-0.1, 0.6, 0.1], [0.2, -0.6, 0.5]]],
                  [[[0.7, 0.1, 0.8], [0.6, -0.7, 0.0], [0.5, 0.9, -0.2]]]])
    conv1 = torch.Tensor(conv1)
    print(conv1.shape)

    conv2 = np.array([[[[0.5]]], [[[0.1]]], [[[0.4]]], [[[-0.1]]], [[[0.6]]], [[[0.1]]],
                      [[[0.2]]], [[[-0.6]]], [[[0.5]]], [[[0.7]]], [[[0.1]]], [[[0.8]]],
                      [[[0.6]]], [[[-0.7]]], [[[0.0]]], [[[0.5]]], [[[0.9]]], [[[-0.2]]]
                      ])
    conv2 = torch.Tensor(conv2)
    print(conv2.shape)

    x = torch.rand(5, 1, 5, 5)
    print(x.shape)

    output1 = F.conv2d(input=x, weight=conv1, stride=(1, 1), padding=1)
    print(output1.shape)

    output2 = F.conv2d(input=x, weight=conv2, stride=(1, 1))
    print(output2.shape)

    s1 = ci_score(output1)
    print(s1)
    s1 = np.mean(s1, axis=0)
    print(s1)

    s2 = ci_score(output2)
    print(s2)
    s2 = np.mean(s2, axis=0)
    print(s2)

def fun15():
    testset = datasets.ImageFolder('./imagedataset')
    print(testset)
    print(testset.classes)

def fun16():
    model = adapter15resnet_34()
    print(model)
    # print(model._modules.items())
    # names = [item[0] for item in model._modules.items()]
    # print(names)

def fun17():
    model = resnet_50([0.] * 100)
    torch.save(model, '../pretrained_models/1.0_resnet_50_imagenet.pth.tar')

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