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
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels


class DTDDataloader(data.Dataset):
    def __init__(self, args, transform=None, train=True):
        classes, class_to_idx = find_classes(os.path.join(args.data_dir, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform

        if train:
            filename = [os.path.join(args.data_dir, 'labels/train' + args.split + '.txt'),
                        os.path.join(args.data_dir, 'labels/val' + args.split + '.txt')]
        else:
            filename = [os.path.join(args.data_dir, 'labels/test' + args.split + '.txt')]

        self.images, self.labels = make_dataset(filename, args.data_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self, args):
        # dtd 32 * 32 mean and std
        # mean: [0.5273, 0.4702, 0.4235] std: [0.2340, 0.2207, 0.2319]
        # dtd 64 * 64 mean and std
        # mean: [0.5273, 0.4702, 0.4235] std: [0.2455, 0.2331, 0.2431]
        # 使用imagenet的mean和std
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.5273, 0.4702, 0.4235], std=[0.2455, 0.2331, 0.2431])
        # transform_train = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ])

        transform_train = transforms.Compose([
            transforms.Resize(36),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            # transforms.Resize([32, 32]),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = DTDDataloader(args, transform_train, train=True)
        testset = DTDDataloader(args, transform_test, train=False)

        kwargs = {'num_workers': 8, 'pin_memory': True}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainloader = trainloader
        self.testloader = testloader

    def getloader(self):
        return self.classes, self.trainloader, self.testloader


def load_dtd_data(args):
    dtd_dataloader = Dataloder(args)
    classes, train_loader, val_loader = dtd_dataloader.getloader()
    return train_loader, val_loader

if __name__ == '__main__':
    # load_cifar_data(None)
    print(torch.__version__)
    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--data_dir', type=str, default='/Users/chenjie/dataset/dtd', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--split', type=str, default='1', help='batch size')
    args = parser.parse_args()

    train_loader, val_loader = load_dtd_data(args)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))



