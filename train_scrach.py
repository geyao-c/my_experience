import os
import time, datetime
import argparse
import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

import utils_append
from models.resnet_cifar import resnet_56, resnet_110, resnet_80, resnet_20, resnet_32
from models.vgg_cifar10 import vgg_16_bn
from models.adapter_resnet_new import adapter1resnet_56, adapter2resnet_56, \
    adapter3resnet_56, adapter5resnet_56, adapter6resnet_56, adapter7resnet_56
from models.resnet_tinyimagenet import resnet_tinyimagenet_56
from models.adapter_resnet_tinyimagenet import adapter3resnet_tinyimagenet_56
from models.adapter_resnet_new_new import adapter8resnet_56
from torch.utils.tensorboard import SummaryWriter
from models.adapter_resnet_new_three import adapter9resnet_56, adapter10resnet_56, \
    adapter11resnet_56, adapter12resnet_56, adapter13resnet_56, adapter14resnet_56, \
    adapter15resnet_56, adapter17resnet_56, adapter16resnet_56, adapter18resnet_56, \
    adapter19resnet_56, adapter20resnet_56, adapter21resnet_56, adapter22resnet_56, \
    adapter23resnet_56, adapter24resnet_56, adapter15resnet_20, adapter16resnet_20, \
    adapter17resnet_20, adapter18resnet_20, adapter19resnet_20, adapter16resnet_32, \
    adapter15resnet_32
from models.sl_mlp_resnet_cifar import sl_mlp_resnet_56
from models.adapter_vgg_cifar10 import adapter_vgg_16_bn, adapter_vgg_16_bn_v2, adapter_vgg_16_bn_v3, adapter_vgg_16_bn_v4
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
from models.mobilenetv2 import mobilenet_v2, mobilenet_v2_change, mobilenet_v2_change_2
from util.focal_loss import FocalLoss
from data import cifar10, cifar100, cub, mnist
from util.losses import SupConLoss
import utils
from thop import profile
import time

def argsget():
    parser = argparse.ArgumentParser("Train scrach")
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--arch', type=str, default='resnet_56', # choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'resnet_50'),
                        help='architecture to calculate feature maps')
    parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
    parser.add_argument('--result_dir', type=str, default='./result/scrach_result56',
                        help='results path for saving models and loggers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    # 这个可以去掉
    # parser.add_argument('--lr_decay_step', default='50,100', type=str, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    # parser.add_argument('--split', type=str, default='1', help='batch size')
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, step, len_iter, args, logger):
    if args.lr_type == 'step':
        # for param_group in optimizer.param_groups:
        #     cur_lr = param_group['lr']
        if epoch >= int(args.epochs * 0.5) and epoch < int(args.epochs * 0.75):
            lr = args.learning_rate * (0.1 ** 1)
        elif epoch >= int(args.epochs * 0.75):
            lr = args.learning_rate * (0.1 ** 2)
        else:
            lr = args.learning_rate

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    # Warmup
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))


def train(epoch, train_loader, model, criterion, optimizer, args, logger, print_freq, device, scheduler=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        adjust_learning_rate(optimizer, epoch, i, num_iter, args, logger)

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args, logger, device):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
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

        logger.info('validate loss {:.3f} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(losses.avg, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def classes_num(args):
    dsetlist = ['cifar10', 'cifar100', 'cub']
    # 最后一层全连接层神经元数量
    clslist = [10, 100, 200]
    idx = dsetlist.index(args.dataset)
    CLASSES = clslist[idx]
    return CLASSES

def dstget(args):
    dsetlist = ['cifar10', 'cifar100', 'cub']
    dldfunlist = [cifar10.load_cifar_data, cifar100.load_cifar_data, cub.load_cub_data]
    idx = dsetlist.index(args.dataset)
    train_loader, val_loader = dldfunlist[idx](args)
    return train_loader, val_loader

def logstore(writer, train_losses, train_accuracy, test_losses, test_accuracy, epoch):
    writer.add_scalar('losses/train losses', train_losses, epoch)
    writer.add_scalar('accuracy/train accuracy', train_accuracy, epoch)
    writer.add_scalar('losses/test losses', test_losses, epoch)
    writer.add_scalar('accuracy/test accuracy', test_accuracy, epoch)

def main():
    # 使用cudnn库加速卷积计算
    cudnn.benchmark = True
    cudnn.enabled = True

    args = argsget()

    # 建立日志
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 当前时间
    args.result_dir = os.path.join(args.result_dir, now)
    logger, writer = utils_append.lgwt_construct(args)
    logger.info("args = %s", args)

    gpuid = 'cuda:{}'.format(args.gpu)
    device = torch.device(gpuid if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    print_freq = (256 * 50) // args.batch_size

    # 加载数据
    train_loader, val_loader = utils_append.dstget(args)
    logger.info('the training dataset is {}'.format(args.dataset))

    # 构建模型
    logger.info('construct model')
    CLASSES = utils_append.classes_num(args.dataset)
    # 创建两个模型一个用来训练，另一个用来计算参数
    if 'adapter' in args.arch:
        model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.]*100,
                                dataset=args.dataset).to(device)
        params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.] * 100,
                                       dataset=args.dataset).to(device)
    else:
        model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, dataset=args.dataset).to(device)
        params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, dataset=args.dataset).to(device)

    logger.info(model)
    if 'tinyimagenet' in args.dataset:
        input_size = 64
    elif 'dtd' == args.dataset:
        input_size = 128
    else:
        input_size = 32
    logger.info('input size: {}'.format(input_size))
    # 计算模型参数量
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model=params_model, device=device, input_size=input_size,
                                                                       dtst=args.dataset)
    logger.info('model flops is {}, params is {}'.format(flops, params))

    # 定义优化器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

    start_epoch = 0
    best_top1_acc = 0

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        # if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        start = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion,
                                                          optimizer, args, logger, print_freq, device)  # , scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args, logger, device)
        logstore(writer, train_obj, train_top1_acc, valid_obj, valid_top1_acc, epoch)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.result_dir)

        epoch += 1
        end = time.time()
        logger.info("=>Best accuracy {:.3f} cost time is {:.3f}".format(best_top1_acc, (end - start)))

if __name__ == '__main__':
    main()