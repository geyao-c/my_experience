import utils_append
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
    fun10()