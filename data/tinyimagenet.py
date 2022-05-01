'''
https://github.com/lmbxmu/HRankPlus
'''
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        pin_memory = True
        # if args.gpu is not None:
        #     pin_memory = True

        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        # imagenet mean and std
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # tinyimagenet mean and std
        # normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        image_size = 64
        # 第一种变换方式
        # train_transform = transforms.Compose([transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(),
        #                                       transforms.ToTensor(), normalize,])
        # test_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize,])

        print('image size is {}'.format(image_size))
        # 第二种变换方式
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = datasets.ImageFolder(traindir, transform=train_transform)
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=4, pin_memory=pin_memory)

        testset = datasets.ImageFolder(valdir, transform=test_transform)
        self.test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True)

def load_tinyimagenet_data(args):
    tinyimagenet = Data(args)
    train_loader, test_loader = tinyimagenet.train_loader, tinyimagenet.test_loader
    return train_loader, test_loader
