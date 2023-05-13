'''
https://github.com/lmbxmu/HRankPlus
'''
import torch
import torch.utils
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
from util.cutout import Cutout

def load_cifar_data(args):
    # imagenet mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # cifar10的mean和std，这里的std有点问题
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 流传的cifar10的std的正确版本
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    # if args.cutout:
    #     transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    # this is a
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader