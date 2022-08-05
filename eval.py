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
import torch.nn.functional as F
from models.resnet_cifar import resnet_56, resnet_110
from models.adapter_resnet_new import adapter1resnet_56, adapter2resnet_56, adapter3resnet_56
from models.adapter_resnet_new_three import adapter15resnet_56
import utils_append
from data import cifar10, cifar100, cub
from data import cifar10
import utils

"""
python eval.py --dataset cifar100 --data_dir ./data --arch adapter15resnet_56 \
--batch_size 128 --pretrain_dir ./pruned_models/71.67_adapter15resnet_56_cifar10tocifar100_pruned_48.pth.tar \
--sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 --adapter_sparsity [0.4]

--sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --adapter_sparsity [0.7]
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

def validate_new(val_loader, model):
    y_pred, y_true = None, None
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if y_pred == None:
                y_pred = logits
                y_true = targets
            else:
                y_pred = torch.cat((y_pred, logits), dim=0)
                y_true = torch.cat((y_true, targets), dim=0)
            print(y_pred.shape)
            print(y_true.shape)
            # if i == 1:
            #     break
    y_pred = F.softmax(y_pred, dim=1)
    acc_original = np.mean([y_pred.argmax(1).numpy() == y_true.numpy()])
    print('acc original: {}'.format(acc_original))
    return y_pred.numpy(), y_true.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("model test")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset test')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--arch', type=str, default='resnet_56', #choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'resnet_50'),
                        help='architecture to calculate feature maps')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain model path')
    parser.add_argument('--sparsity', type=str, default=None, help='sparsity of each conv layer')
    parser.add_argument('--adapter_sparsity', type=str, default=None, help='sparsity of each adapter layer')

    args = parser.parse_args()

    # 加载数据
    train_loader, val_loader = utils_append.dstget(args)
    # 构建模型
    CLASSES = utils_append.classes_num(args.dataset)

    # 解析adapter sparsity
    if args.adapter_sparsity:
        adapter_sparsity = utils_append.analysis_sparsity(args.adapter_sparsity)
    elif 'adapter' in args.finetune_arch:
        raise ValueError('adapter sparsity is None')

    # 解析sparsity
    if args.sparsity:
        sparsity = utils_append.analysis_sparsity(args.sparsity)
    else:
        raise ValueError('sparsity is None')

    if 'adapter' in args.arch:
        # model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.]*100).to(device)
        model = eval(args.arch)(sparsity=sparsity, num_classes=CLASSES, adapter_sparsity=adapter_sparsity).to(device)
    else:
        # model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES).to(device)
        model = eval(args.arch)(sparsity=sparsity, num_classes=CLASSES).to(device)
        # params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES).to(device)
    mapstr = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.pretrain_dir, map_location=mapstr)
    model.load_state_dict(ckpt['state_dict'], strict=False)

    # 评价器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # 验证准确率
    # valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
    y_pred, y_true = validate_new(val_loader, model)
    # print('before can accuracy: {}'.format(valid_top1_acc))

    # 评价每个预测结果的不确定性
    k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
    y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)
    print(y_pred_topk.shape)
    print(y_pred_topk[0])

    threshold = 0.9
    y_pred_confident= y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
    y_true_confident = y_true[y_pred_uncertainty < threshold]
    y_true_unconfident = y_true[y_pred_uncertainty >= threshold]

    acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
    acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
    print('confident acc: {}'.format(acc_confident))
    print('unconfident acc: {}'.format(acc_unconfident))

    if args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    prior = np.zeros(num_classes)
    for d in train_loader.dataset:
        prior[d[1]] += 1
    print(prior.shape)
    print(prior.sum())
    prior /= prior.sum()

    right, alpha, iters = 0, 1, 1
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1
    acnum = acc_confident * len(y_pred_confident)
    print('acnum: {}, right: {}'.format(acnum, right))
    acc_final = (acnum + right) / len(y_pred)
    print('new unconfident acc is {}'.format(right / (i + 1.)))
    print('final acc {}'.format(acc_final))

