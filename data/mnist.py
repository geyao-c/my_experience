'''
https://github.com/lmbxmu/HRankPlus
'''
import torch
import torch.utils
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms

def load_cifar_data(args):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader