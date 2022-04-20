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

def load_flowers_data(args):
    # imagenet mean and std
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # cifar10的mean和std，这里的std有点问题
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 流传的cifar10的std的正确版本
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))

    # data_dir = '/Users/chenjie/dataset/flowers102_splited'

    # TODO: Define your transforms for the training, validation, and testing sets
    # defining data transforms for training, validation and test data and also normalizing whole data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # TODO: Load the datasets with ImageFolder
    # loading datasets with PyTorch ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # defining data loaders to load data using image_datasets and transforms, here we also specify batch size for the mini batch
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=True, num_workers=1)

    class_names = image_datasets['train'].classes

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # load_cifar_data(None)
    # print(torch.__version__)
    print(torch.__version__)
    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--data_dir', type=str, default='/Users/chenjie/dataset/flowers102_splited', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('--split', type=str, default='1', help='batch size')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_flowers_data(args)
    print(len(train_loader.dataset))
    for x in train_loader:
        print(x[1])
        break
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))