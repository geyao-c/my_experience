import torchvision.utils as vutils
import torchvision, torch
from torchvision import transforms
from data import cifar10, cifar100
from utils import TwoCropTransform
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    # trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=TwoCropTransform(transform_train))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    data_iter = iter(train_loader)

    carimages1 = []
    carimages2 = []
    while 1:
        images = data_iter.next()
        # print(len(images))
        print(images[1])
        print(images[0][0].shape)
        n = images[0][0].shape[0]
        for i in range(n):
            if images[1][i] == 1:
                carimages1.append(images[0][0][i])
                # carimages2.append(images[1][0][i])
            if len(carimages1) == 16: break
        if len(carimages1) == 16: break


    # vutils.save_image(carimages, './saved_images/grid_image.jpg', nrow=4, padding=0)
    # vutils.save_image(carimages1, './saved_images/image2_1.jpg', nrow=4, padding=0)
    vutils.save_image(carimages1, './saved_images/image2_2.jpg', nrow=4, padding=0)

