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
from util.losses import SupConLoss
from models.selfsupcon_supcon_adapter_resnet import selfsupcon_supcon_adapter15resnet_56, selfsupcon_supcon_resnet_56, \
    selfsupcon_supcon_adapter15resnet_20
from data import cifar10, cifar100
import utils
import numpy as np
from thop import profile
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argsget():
    parser = argparse.ArgumentParser("Train scrach")
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used')
    parser.add_argument('--arch', type=str, default='resnet_56', # choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'resnet_50'),
                        help='architecture to calculate feature maps')
    parser.add_argument('--lr_decay_epochs', type=str, default='100, 150',
                        help='where to decay lr, can be a list')
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
    parser.add_argument('--split', type=str, default='1', help='batch size')
    parser.add_argument('--supcontemp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--selfsupcontemp', type=float, default=0.07, help='temperature for loss function')
    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--selfsupconlossxs', type=float, default=1.0)
    parser.add_argument('--supconlossxs', type=float, default=1.0)
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, step, len_iter, args, logger):
    warmup_epoch = 10
    if args.lr_type == 'step':
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = args.learning_rate * (0.1 ** steps)
        else:
            lr = args.learning_rate

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        # lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
        lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - warmup_epoch) / (args.epochs - warmup_epoch)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    # Warmup

    if epoch < warmup_epoch:
        # lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)
        lr = lr * float(1 + step + epoch * len_iter) / (warmup_epoch * len_iter)


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))


def train(epoch, train_loader, model, criterion, optimizer, args, logger, print_freq, supcon_criterion=None,
          selfsupcon_criterion=None, scheduler=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    supcon_losses = utils.AverageMeter('Loss', ':.4e')
    selfsupcon_losses = utils.AverageMeter('Loss', ':.4e')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    # adjust_learning_rate(optimizer, epoch, 0, num_iter)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(device)
        target = target.to(device)

        bsz = target.shape[0]
        adjust_learning_rate(optimizer, epoch, i, num_iter, args, logger)

        # compute outputy
        selfsupcon_logits, supcon_logits = model(images)
        # ce_logits, supcon_logits = model(images)
        supconf1, supconf2 = torch.split(supcon_logits, [bsz, bsz], dim=0)
        selfsupconf1, selfsupconf2 = torch.split(selfsupcon_logits, [bsz, bsz], dim=0)
        psupcon_logits = torch.cat([supconf1.unsqueeze(1), supconf2.unsqueeze(1)], dim=1)
        pselfsupcon_logits = torch.cat([selfsupconf1.unsqueeze(1), selfsupconf2.unsqueeze(1)], dim=1)

        supcon_loss = args.supconlossxs * supcon_criterion(psupcon_logits, target)
        selfsupcon_loss = args.selfsupconlossxs * selfsupcon_criterion(pselfsupcon_logits)

        loss =  selfsupcon_loss +  supcon_loss

        supcon_losses.update(supcon_loss.item(), bsz)
        selfsupcon_losses.update(selfsupcon_loss.item(), bsz)
        losses.update(loss.item(), bsz)  # accumulated loss

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
                'supcon loss is {supcon_loss.avg:.4f}, selfsupcon_loss is {selfsupcon_loss.avg:.4f}, Loss {loss.avg:.4f}'
                .format(epoch, i, num_iter, supcon_loss=supcon_losses, selfsupcon_loss=selfsupcon_losses, loss=losses))

    # return losses.avg, supcon_losses.avg, ce_losses.avg, top1.avg, top5.avg
    return losses.avg, supcon_losses.avg, selfsupcon_losses.avg


def validate(epoch, val_loader, model, criterion, args, logger):
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
            logits, _ = model(images)
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


# coding=utf-8

import shlex
import datetime
import subprocess
import time

def isok(sub_list):
    for item in sub_list:
        if item.poll() is None:
            return False
    return True

def execute_command(cmdstring_list, cwd=None, timeout=None, shell=True):
    """执行一个SHELL命令
        封装了subprocess的Popen方法, 支持超时判断，支持读取stdout和stderr
        参数:
      cwd: 运行命令时更改路径，如果被设定，子进程会直接先更改当前路径到cwd
      timeout: 超时时间，秒，支持小数，精度0.1秒
      shell: 是否通过shell运行
    Returns: return_code
    Raises: Exception: 执行超时
    """
    # if shell:
    #     cmdstring_list = cmdstring
    # else:
    #     cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    sub_list = []
    # 没有指定标准输出和错误输出的管道，因此会打印到屏幕上；
    for i, item in enumerate(cmdstring_list):
        sub = subprocess.Popen(item, cwd=cwd, stdin=subprocess.PIPE, shell=shell, bufsize=4096)
        sub_list.append(sub)

    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中
    print('开始执行')
    while True:
        if isok(sub_list) is True: break
        time.sleep(0.5)
    print('执行完了')
    # return str(sub.returncode)

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

    print_freq = (256 * 50) // args.batch_size

    # 加载数据
    # train_loader, val_loader = utils_append.dstget(args)
    train_loader, val_loader = utils_append.supcon_dstget(args)
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
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model=params_model, device=device, input_size=input_size)
    logger.info('model flops is {}, params is {}'.format(flops, params))

    # 定义优化器
    supcon_criterion = SupConLoss(temperature=args.supcontemp)
    supcon_criterion.to(device)
    selfsupcon_criterion = SupConLoss(temperature=args.selfsupcontemp)
    selfsupcon_criterion.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

    start_epoch = 0
    best_top1_acc = 0

    # iterations = args.lr_decay_epochs.split(',')
    # args.lr_decay_epochs = list([])
    # for it in iterations:
    #     args.lr_decay_epochs.append(int(it))
    # logger.info('lr_decay_epochs is : {}'.format(args.lr_decay_epochs))
    logger.info('method is {}'.format(args.method))
    # train the model
    epoch = start_epoch
    logger.info("device is {}".format(device))

    jieduan_ckpt_dir = os.path.join('./pretrained_models', 'selfsupcon-supcon_scrach_train', args.arch, now)
    if not os.path.exists(jieduan_ckpt_dir):
        os.makedirs(jieduan_ckpt_dir)

    while epoch < args.epochs:
        start = time.time()
        total_loss, supcon_loss, selfsupcon_loss = train(epoch, train_loader, model, criterion, optimizer, args,
                          logger, print_freq, supcon_criterion=supcon_criterion, selfsupcon_criterion=selfsupcon_criterion)  # , scheduler)
        total_loss, supcon_loss, selfsupcon_loss = round(total_loss, 2), round(supcon_loss, 2), round(selfsupcon_loss, 2)
        # valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args, logger)
        utils_append.lossstore(writer, train_losses=total_loss, selfsupcon_losses=selfsupcon_loss,
                               supcon_losses=supcon_loss, epoch=epoch)
        # is_best = False
        # if valid_top1_acc > best_top1_acc:
        #     best_top1_acc = valid_top1_acc
        #     is_best = True

        logger.info('epoch {}, total_loss is {:.2f}, supcon_loss is {:.2f}, selfsupcon_loss is {:.2f}'.
                    format(epoch, total_loss, supcon_loss, selfsupcon_loss))
        # epochlist = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        epochlist = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        k = 12
        if (epoch + 1) in epochlist:
            # 保存模型
            jieduan_ckpt_model_name = '{}_epoch{}_{}_{}.pth.tar'.format(total_loss,
                                            epoch + 1, args.arch, args.dataset)
            jieduan_ckpt_path = os.path.join(jieduan_ckpt_dir, jieduan_ckpt_model_name)
            utils.save_jieduan_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':
                optimizer.state_dict(),}, jieduan_ckpt_path)

            # 计算conv feature map
            # model_accu + '_' + args.arch + '_' + args.dataset + '_repeat%d' % (args.repeat)
            # fmap_saved_dir = os.path.join("../conv_feature_map", 'selfsupcon-supcon_scrach_train', now,
            #                               str(total_loss) + '_' + args.arch + '_' + args.dataset + '_repeat%d' % (5))
            fmap_saved_dir = os.path.join("../conv_feature_map", 'selfsupcon-supcon_scrach_train', "k={}".format(k), args.arch, now,
                                          str(total_loss) + '_' + args.dataset + '_repeat%d' % (5))
            fmap_cal_cmd = "python calculate_feature_maps_n.py --arch {} --dataset {} --data_dir ./data " \
                           "--pretrain_dir {} --save_dir {}".format(args.arch, args.dataset, jieduan_ckpt_path, fmap_saved_dir)
            print(fmap_cal_cmd)
            execute_command([fmap_cal_cmd])

            # 计算selfsupcon_supcon_adapter15resnet_56的ci
            ci_save_dir = os.path.join('./calculated_ci', 'selfsupcon-supcon_scrach_train', "k={}".format(k), now, str(total_loss) + '_' +
                                       args.arch + '_' + args.dataset)
            ci_cal_cmd = 'python calculate_ci_n.py --arch {} --repeat 5 --num_layers 55 \
            --feature_map_dir {} --save_dir {}'.format(args.arch, fmap_saved_dir, ci_save_dir)
            execute_command([ci_cal_cmd])

            # 计算selfsupcon_supcon_adapter15resnet_20的ci
            # ci_save_dir = os.path.join('./calculated_ci', 'selfsupcon-supcon_scrach_train', args.arch, now, str(total_loss) +
            #                             '_' + args.dataset)
            # ci_cal_cmd = 'python calculate_ci_n.py --arch {} --repeat 5 --num_layers 19 \
            #             --feature_map_dir {} --save_dir {}'.format(args.arch, fmap_saved_dir, ci_save_dir)
            # execute_command([ci_cal_cmd])
            # 压缩迁移微调，从cifar10迁移到svhn
            # graf_pruned_60_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', args.arch, now,
            #                                          str(total_loss) + '_' + 'cifar10tosvhn' + '_' +
            #                                          'adapter15resnet_20' + '_' + 'pruned_60')
            # graf_pruned_60_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
            #                      "svhn --finetune_data_dir ./data/svhn --pretrained_arch adapter15resnet_20 --finetune_arch adapter15resnet_20 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.01 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.3]*2+[0.4]*3+[0.5]*3+[0.6]*3 --adapter_sparsity [0.6]" \
            #                      "".format(graf_pruned_60_result_dir, ci_save_dir, jieduan_ckpt_path)
            # print("graf_pruned_60_cmd: ", graf_pruned_60_cmd)

            # 压缩迁移微调，从cifar10到cifar10
            # graf_pruned_60_result_dir_normal = os.path.join('./result/selfsupcon-supcon_graf_pruned_normal', args.arch, now,
            #                                          str(total_loss) + '_' +
            #                                          'cifar10tocifar10' + '_' + 'adapter15resnet_20' + '_' + 'pruned_60')
            # graf_pruned_60_cmd_normal = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
            #                      "cifar10 --finetune_data_dir ./data --pretrained_arch adapter15resnet_20 --finetune_arch adapter15resnet_20 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.01 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.3]*2+[0.4]*3+[0.5]*3+[0.6]*3 --adapter_sparsity [0.6]" \
            #                             "".format(graf_pruned_60_result_dir_normal, ci_save_dir, jieduan_ckpt_path)

            # 压缩迁移微调，从cifar10到cifar10, 裁剪率70%
            graf_pruned_70_result_dir_normal = os.path.join('./result/selfsupcon-supcon_graf_pruned_normal', "k={}".format(k), args.arch,
                                                            now,
                                                            str(total_loss) + '_' +
                                                            'cifar10tocifar10' + '_' + 'adapter15resnet_56' + '_' + 'pruned_70')
            graf_pruned_70_cmd_normal = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
                                        "cifar10 --finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 " \
                                        "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.05 " \
                                        "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --adapter_sparsity [0.75]" \
                                        "".format(graf_pruned_70_result_dir_normal, ci_save_dir, jieduan_ckpt_path)

            # graf_pruned_48_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', now, str(total_loss) + '_' +
            #                                          'cifar10tocifar100' + '_' + 'adapter15resnet_56' + '_' + 'pruned_48')
            # graf_pruned_48_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', now,
            #                                          str(total_loss) + '_' +
            #                                          'cifar10tocifar100' + '_' + 'resnet_56' + '_' + 'pruned_48')

            # graf_pruned_48_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', now,
            #                                          str(total_loss) + '_' +
            #                                          'cifar100tocifar10' + '_' + 'adapter15resnet_56' + '_' + 'pruned_48')

            # graf_pruned_48_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
            #                      "cifar100 --finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 " \
            #                      "--adapter_sparsity [0.4]".format(graf_pruned_48_result_dir, ci_save_dir, jieduan_ckpt_path)
            # graf_pruned_48_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
            #                      "cifar100 --finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9" \
            #                      "".format(graf_pruned_48_result_dir, ci_save_dir, jieduan_ckpt_path)

            # graf_pruned_48_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset " \
            #                      "cifar10 --finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 " \
            #                      "--adapter_sparsity [0.4]".format(graf_pruned_48_result_dir, ci_save_dir,
            #                                                        jieduan_ckpt_path)

            graf_pruned_70_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', "k={}".format(k), args.arch,
                                                     now, str(total_loss) + '_' +
                                                     'cifar10tocifar100' + '_' + 'adapter15resnet_56' + '_' + 'pruned_70')
            # graf_pruned_70_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', now, str(total_loss) + '_' +
            #                                          'cifar10tocifar100' + '_' + 'resnet_56' + '_' + 'pruned_70')


            # graf_pruned_70_result_dir = os.path.join('./result/selfsupcon-supcon_graf_pruned', now,
            #                                          str(total_loss) + '_' +
            #                                          'cifar100tocifar100' + '_' + 'adapter15resnet_56' + '_' + 'pruned_70')


            graf_pruned_70_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
                                 "cifar100 --finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 " \
                                 "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
                                 "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 " \
                                 "--adapter_sparsity [0.75]".format(graf_pruned_70_result_dir, ci_save_dir, jieduan_ckpt_path)
            # graf_pruned_70_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset " \
            #                      "cifar100 --finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9" \
            #                      "".format(graf_pruned_70_result_dir, ci_save_dir, jieduan_ckpt_path)


            # graf_pruned_70_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset " \
            #                      "cifar10 --finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 " \
            #                      "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
            #                      "--momentum 0.9 --weight_decay 0.0005 --pretrain_dir {} --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 " \
            #                      "--adapter_sparsity [0.7]".format(graf_pruned_70_result_dir, ci_save_dir,
            #                                                        jieduan_ckpt_path)
            # execute_command([graf_pruned_60_cmd, graf_pruned_60_cmd_normal])
            execute_command([graf_pruned_70_cmd, graf_pruned_70_cmd_normal])

        utils.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':
                                optimizer.state_dict(),}, False, args.result_dir)
        epoch += 1
        end = time.time()
        logger.info("=>Best accuracy {:.3f} cost time is {:.3f}".format(best_top1_acc, (end - start)))


if __name__ == '__main__':
    main()
