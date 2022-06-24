import copy
import os
import time, datetime
import argparse
import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import shlex
import datetime
import subprocess
import time

import utils_append
from models.resnet_cifar import resnet_56, resnet_110, resnet_80
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
    adapter23resnet_56, adapter24resnet_56
from models.sl_mlp_resnet_cifar import sl_mlp_resnet_56
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
from util.focal_loss import FocalLoss
from data import cifar10, cifar100, cub
from util.losses import SupConLoss
import utils
from thop import profile
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argsget():
    parser = argparse.ArgumentParser("Train pruned pipline")
    # train scrach的数据集合数据集文件夹
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset')
    # 使用的网络结构
    parser.add_argument('--arch', type=str, default='resnet_56', help='architecture to calculate feature maps')
    # 学习率类型，train scarch默认为step
    parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
    parser.add_argument('--result_dir', type=str, default='./result/scrach_train/adapter15resnet56_cifar10',
                        help='results path for saving models and loggers')
    # 训练参数设置
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, step, len_iter, args, logger):
    if args.lr_type == 'step':
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        # factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        # lr = args.learning_rate * (0.1 ** factor)

        # factor = epoch // 125
        # if epoch in [args.epochs*0.5, args.epochs*0.75]:
        lr = cur_lr
        if epoch in [80, 120]:
            lr = cur_lr / 10
        # lr = args.learning_rate * (0.1 ** factor)

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
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))


def train(epoch, train_loader, model, criterion, optimizer, args, logger, print_freq, scheduler=None):
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
    # adjust_learning_rate(optimizer, epoch, 0, num_iter)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # adjust_learning_rate(optimizer, epoch, i, num_iter, args, logger)

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

    # scheduler.step()

    return losses.avg, top1.avg, top5.avg


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
        time.sleep(5)
        sub_list.append(sub)

    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中
    print('开始执行')
    while True:
        if isok(sub_list) is True: break
        time.sleep(0.5)
    print('执行完了')
    # return str(sub.returncode)

def pruned_finetune_pipline(args, now, accu, model):
    jieduan_ckpt_dir = os.path.join('./pretrained_models', '{}_scrach_train_{}'.format(args.arch, args.dataset), now)
    if not os.path.exists(jieduan_ckpt_dir):
        os.makedirs(jieduan_ckpt_dir)
    if accu >= 93.59 and accu <= 93.73:
    # if accu >= 0.0 and accu <= 100:
        jieduan_ckpt_model_name = '{}_{}_{}.pth.tar'.format(accu, args.arch, args.dataset)
        jieduan_ckpt_path = os.path.join(jieduan_ckpt_dir, jieduan_ckpt_model_name)
        utils.save_jieduan_checkpoint({'state_dict': model.state_dict()}, jieduan_ckpt_path)

        # 计算feature map
        fmap_saved_dir = os.path.join("../conv_feature_map", '{}_scrach_train_{}'.format(args.arch, args.dataset), now,
                                      str(accu) + '_' + args.arch + '_' + args.dataset + '_repeat%d' % (5))
        fmap_cal_cmd = "python calculate_feature_maps_n.py --arch {} --dataset {} --data_dir {} --pretrain_dir {} " \
                       "--save_dir {}".format(args.arch, args.dataset, args.data_dir, jieduan_ckpt_path, fmap_saved_dir)
        print(fmap_cal_cmd)
        execute_command([fmap_cal_cmd])

        # 计算ci
        ci_save_dir = os.path.join('./calculated_ci', '{}_scrach_train_{}'.format(args.arch, args.dataset), now,
                                   str(accu) + '_' + args.arch + '_' + args.dataset)
        ci_cal_cmd = 'python calculate_ci_n.py --arch {} --repeat 5 --num_layers 55 --feature_map_dir {} ' \
                     '--save_dir {}'.format(args.arch, fmap_saved_dir, ci_save_dir)
        print(ci_cal_cmd)
        execute_command([ci_cal_cmd])

        # 压缩并微调
        graf_pruned_70_result_dir = os.path.join('./result/normal_pruned', now, str(accu) + '_' + '{}to{}'.format(
                                    args.dataset, args.dataset) + '_' + '{}'.format(args.arch) + '_' + 'pruned_70')

        graf_pruned_70_cmd = "python graf_prune_finetune_cifar_n.py --pretrained_dataset {} --finetune_dataset " \
                             "{} --finetune_data_dir {} --pretrained_arch {} --finetune_arch {} " \
                             "--result_dir {} --ci_dir {} --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 " \
                             "--momentum 0.9 --weight_decay 0.0005 --graf --pretrain_dir {} --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 " \
                             "--adapter_sparsity [0.7]".format(args.dataset, args.dataset, args.data_dir, args.arch, args.arch,
                                                               graf_pruned_70_result_dir, ci_save_dir, jieduan_ckpt_path)
        execute_command([graf_pruned_70_cmd, graf_pruned_70_cmd])

def main():
    # 使用cudnn库加速卷积计算
    cudnn.benchmark = True
    cudnn.enabled = True

    args = argsget()
    mid_result_dir = args.result_dir

    while 1:
        logger, writer = None, None

        # 建立日志
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 当前时间
        args.result_dir = os.path.join(mid_result_dir, now)
        logger, writer = utils_append.lgwt_construct(args)
        logger.info("args = %s", args)

        print_freq = (256 * 50) // args.batch_size

        # 加载数据
        train_loader, val_loader = utils_append.dstget(args)
        logger.info('the training dataset is {}'.format(args.dataset))

        # 构建模型
        logger.info('construct model')
        CLASSES = utils_append.classes_num(args.dataset)
        # 创建两个模型一个用来训练，另一个用来计算参数
        if 'adapter' in args.arch:
            model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.]*100, dataset=args.dataset).to(device)
            params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, adapter_sparsity=[0.] * 100, dataset=args.dataset).to(device)
        else:
            model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, dataset=args.dataset).to(device)
            params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=CLASSES, dataset=args.dataset).to(device)

        # logger.info(model)
        input_size = 32
        logger.info('input size: {}'.format(input_size))

        # 计算模型参数量
        flops, params, flops_ratio, params_ratio = utils_append.cal_params(model=params_model, device=device, input_size=input_size)
        logger.info('model flops is {}, params is {}'.format(flops, params))

        # 设置损失，默认使用交叉山损失函数
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # 定义优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)

        start_epoch = 0
        best_top1_acc = 0

        # 训练模型
        epoch = start_epoch
        best_accu_model, valid_top1_acc = None, None

        while epoch < args.epochs:
            # 学习率在0.5和0.75的时候乘以0.1
            if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            start = time.time()
            train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion, optimizer, args, logger, print_freq)
            valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args, logger)
            logstore(writer, train_obj, train_top1_acc, valid_obj, valid_top1_acc, epoch)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                best_accu_model = copy.copy(model)
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
        logger.info("best top1 accu is {}, valid top1 accu is {}".format(best_top1_acc, valid_top1_acc))
        best_top1_acc = round(best_top1_acc.item(), 2)
        valid_top1_acc = round(valid_top1_acc.item(), 2)
        # 训练完成后会有两个模型精度，一个是最后一个模型，一个是验证集上精度最好的一个模型
        # 在这个精度范围内的模型才进行后续裁剪和微调操作
        # pruned_finetune_pipline(args, now, best_top1_acc, best_accu_model)
        # pruned_finetune_pipline(args, now, valid_top1_acc, model)

if __name__ == '__main__':
    main()