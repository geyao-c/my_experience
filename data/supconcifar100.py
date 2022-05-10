'''
https://github.com/lmbxmu/HRankPlus
'''
import torch
import torch.utils
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
from util.cutout import Cutout
from utils import TwoCropTransform

def load_cifar_data(args):
    # imagenet mean and std
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # cifar10的mean和std，这里的std有点问题
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 流传的cifar10的std的正确版本
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # if args.cutout:
    #     transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    # this is a
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=TwoCropTransform(transform_train))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader