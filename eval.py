import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models.resnet_cifar import resnet_56, resnet_110
from models.adapter_resnet_new import adapter1resnet_56, adapter2resnet_56, adapter3resnet_56
import utils_append
from data import cifar10, cifar100, cub
from data import cifar10
import utils

"""
python eval.py --dataset cifar100 --data_dir ./data --arch adapter3resnet_56 \
--batch_size 128 --pretrain_dir ./pretrained_models/74.19_adapter4resnet_56_cifar100.pth.tar
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    count = 0
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            count += images.shape[0]
            print(count, '/', len(val_loader.dataset))
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser("model test")
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset test')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--arch', type=str, default='resnet_56', #choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'resnet_50'),
                        help='architecture to calculate feature maps')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain model path')
    args = parser.parse_args()

    # 加载数据
    train_loader, val_loader = utils_append.dstget(args)
    # 构建模型
    CLASSES = utils_append.classes_num(args.dataset)

    if 'adapter' in args.arch:
        model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.]*100).to(device)
        # params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.] * 100).to(device)
    else:
        model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES).to(device)
        # params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES).to(device)
    mapstr = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.pretrain_dir, map_location=mapstr)
    model.load_state_dict(ckpt['state_dict'])

    # 评价器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # 验证准确率
    valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
    print('accuracy: {}'.format(valid_top1_acc))



