'''
https://github.com/lmbxmu/HRankPlus
'''
import torch
import torch.utils
import torch.utils.data.distributed

import torchvision
import torch
from torchvision import datasets, transforms
import argparse
import os

def load_svhn_data(args):
    # imagenet mean and std
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # cifar10的mean和std，这里的std有点问题
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 流传的cifar10的std的正确版本
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    # svhn的mean和std
    normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # this is a
    trainset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='train', download=True, transform=transform_train)
    # trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='test', download=True, transform=transform_test)
    # testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader

if __name__ == '__main__':
    # load_cifar_data(None)
    print(torch.__version__)
    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--data_dir', type=str, default='./svhn', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()

    load_svhn_data(args)



