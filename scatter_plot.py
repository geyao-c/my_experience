import numpy as np
import torchvision
from torchvision import transforms
from models.resnet_cifar import resnet_56
from models.adapter_resnet_new_three import adapter15resnet_56, adapter21resnet_56
from models.supcon_adapter_resnet import supcon_adapter15resnet_56, selfsupcon_adapter15resnet_56
from models.selfsupcon_supcon_adapter_resnet import selfsupcon_supcon_adapter15resnet_56
import torch
from data import cifar10, cifar100
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(modelpath, args):
    # 加载模型
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100

    model = None
    print('args arc is: {}'.format(args.arc))
    if args.arc == 'adapter15resnet_56':
        model = adapter15resnet_56([0.]*100, num_classes, [0.]*100)
    elif args.arc == 'resnet_56':
        model = resnet_56([0.]*100, num_classes)
    elif args.arc == 'selfsupcon_adapter15resnet_56':
        model = selfsupcon_adapter15resnet_56([0.]*100, num_classes, [0.]*100)
    elif args.arc == 'supcon_adapter15resnet_56':
        model = supcon_adapter15resnet_56([0.]*100, num_classes, [0.]*100)
    elif args.arc == 'selfsupcon_supcon_adapter15resnet_56':
        model = selfsupcon_supcon_adapter15resnet_56([0.]*100, num_classes, [0.]*100)

    map_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(modelpath, map_location=map_str)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    return model

def model_val(model, val_loader, args):
    val_features = None
    val_targets = None
    tot, count = len(val_loader.dataset), 0
    print(tot)
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            count += images.shape[0]
            print('{}/ {}'.format(count, tot))
            images = images.to(device)
            target = target.to(device)
            if args.arc == 'resnet_56':
                logits, feature = model(images)
            elif args.arc == 'adapter15resnet_56':
                print('logits feature')
                logits, feature = model(images)
            elif args.arc == 'selfsupcon_adapter15resnet_56':
                print('selfsupcon logits feature')
                logits, feature = model(images)
            elif args.arc == 'supcon_adapter15resnet_56':
                print('supcon logits feature')
                logits, feature = model(images)
            elif args.arc == 'selfsupcon_supcon_adapter15resnet_56':
                print('selfsupcon-supcon logits features')
                _, _, feature = model(images)
            feature, target = feature.numpy(), target.numpy()
            if val_features is None:
                val_features, val_targets = feature, target
            else:
                val_features = np.vstack((val_features, feature))
                val_targets = np.hstack((val_targets, target))
    return val_features, val_targets

if __name__ == '__main__':

    parser = argparse.ArgumentParser("CIFAR prune training")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    # parser.add_argument('--arc', type=str, default='resnet_56', help='arcs')
    # parser.add_argument('--arc', type=str, default='adapter15resnet_56', help='arcs')
    # parser.add_argument('--arc', type=str, default='selfsupcon_adapter15resnet_56', help='arcs')
    parser.add_argument('--arc', type=str, default='selfsupcon_supcon_adapter15resnet_56', help='arcs')
    # parser.add_argument('--arc', type=str, default='supcon_adapter15resnet_56', help='arcs')

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        train_loader, val_loader = cifar10.load_cifar_data(args)
    elif args.dataset == 'cifar100':
        train_loader, val_loader = cifar100.load_cifar_data(args)

    # name = '4.30_supcon-ce_adapter15resnet_56_cifar10'
    # name = '7.99_supcon-ce_adapter15resnet_56_cifar10'
    # name = '43.46_selfsupcon-ce_adapter15resnet_56_cifar100'
    name = '3.15-1_supcon-ce_adapter15resnet_56_cifar100'
    # name = '41.80_selfsupcon-ce_adapter15resnet_56_cifar10'
    # name = '6.31_supcon_adapter15resnet_56_cifar100'
    # name = '41.93_selfsupcon-ce_adapter15resnet_56_cifar10'
    # name = '0.57_selfsupcon-ce_adapter15resnet_56_cifar100'
    # model1 = get_model('./pretrained_models/33.39_supcon-ce_adapter15resnet_56_cifar10.pth.tar')
    # name = "72.68_resnet_56_cifar100"
    # name = "72.23_resnet_56_cifar100"
    # name = "49.48_epoch1000_selfsupcon-supcon_adapter15resnet_56_cifar100"
    # name = "48.77_epoch1000_selfsupcon_supcon_adapter15resnet_56_cifar100"
    # name = "50.07_epoch1000_selfsupcon-supcon_adapter15resnet_56_cifar10"
    # name = "50.94_epoch800_selfsupcon-supcon_adapter15resnet_56_cifar10"
    # name = "51.65_epoch600_selfsupcon-supcon_adapter15resnet_56_cifar10"
    # name = "60.4_epoch600_selfsupcon_supcon_adapter15resnet_56_cifar10"
    # name = "58.04_epoch1000_selfsupcon_supcon_adapter15resnet_56_cifar10"
    # name = "38.52_epoch700_selfsupcon_supcon_adapter15resnet_56_cifar10"
    name = "37.02_epoch1000_selfsupcon_supcon_adapter15resnet_56_cifar10"
    # name = "94.55_adapter15resnet_56_cifar10"
    # name = "51.55_epoch650_selfsupcon_supcon_adapter15resnet_56_cifar10"
    # name = "50.07_epoch1000_selfsupcon_supcon_adapter15resnet_56_cifar10_pipline"
    # name = "51.92_epoch400_selfsupcon_supcon_adapter15resnet_56_cifar100"
    # name = "93.77_adapter15resnet_56_cifar10"
    model1 = get_model('./pretrained_models/' + name + '.pth.tar', args)
    # model1 = get_model('./pretrained_models/4.30_supcon-ce_adapter15resnet_56_cifar10.pth.tar')
    # model2 = get_model('./pretrained_models/94.54_resnet_56_cifar10.pth.tar')

    val_features1, val_targets1 = model_val(model1, val_loader, args)
    # val_features1, val_targets1 = model_val(model1, train_loader, args)
    # val_features2, val_targets2 = model_val(model2, val_loader)

    # val_features1, val_targets1 = model_val(model1, train_loader)
    # val_features2, val_targets2 = model_val(model2, train_loader)

    X_tsne1 = TSNE(n_components=2).fit_transform(val_features1)
    # X_tsne2 = TSNE(n_components=2).fit_transform(val_features2)

    print(type(X_tsne1))
    cs_xtsne1, cs_xtsne2 = [], []
    cs_tg1, cs_tg2 = [], []
    cs_num = [i for i in range(10)]
    # cs_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # cs_num = [9]
    # cs_num = [7, 8, 9]
    for idx in range(len(X_tsne1)):
        if val_targets1[idx] in cs_num and len(cs_xtsne1) <= 10000:
            cs_xtsne1.append(X_tsne1[idx].tolist())
            cs_tg1.append(val_targets1[idx])
        # if val_targets2[idx] in cs_num and len(cs_xtsne2) <= 10000:
        #     cs_xtsne2.append(X_tsne2[idx].tolist())
        #     cs_tg2.append(val_targets2[idx])
    cs_xtsne1, cs_xtsne2 = np.array(cs_xtsne1), np.array(cs_xtsne2)
    print(cs_tg1)
    print(cs_tg2)
    # 只绘制三类
    plt.figure(figsize=(5, 5))

    # plt.subplot(121)
    # plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], c=val_targets1, label="new", s=1)
    ax = plt.scatter(cs_xtsne1[:, 0], cs_xtsne1[:, 1], c=cs_tg1, label="new", s=1)
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    plt.legend()

    # plt.subplot(122)
    # plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=val_targets2, label="original", s=1)
    # ax = plt.scatter(cs_xtsne2[:, 0], cs_xtsne2[:, 1], c=cs_tg2, label="original", s=1)
    # plt.xlim(-90, 90)
    # plt.ylim(-90, 90)

    plt.legend()
    plt.savefig('images/' + name + '.png', dpi=600)
    plt.show()



