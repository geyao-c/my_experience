import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import datetime
import os
import utils_append
import utils
import time
import math
from data import imagenet
from models.resnet_imagenet import resnet_34
from models.adapter_resnet_imagenet import adapter15resnet_34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先解析参数
parser = argparse.ArgumentParser("ImageNet prune training")
# 训练数据集
parser.add_argument('--pretrained_dataset', type=str, default='imagenet', help='model train dataset used')
# 微调数据集
parser.add_argument('--finetune_dataset', type=str, default='cifar100', help='model train dataset used')
parser.add_argument('--dataset', type=str, default=None, help='model train dataset used')
parser.add_argument('--graf', action="store_false", help='graf pruned or not')
# 微调数据集文件路径
parser.add_argument('--finetune_data_dir', type=str, default='./data', help='path to dataset')
parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
parser.add_argument('--pretrained_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of pretrained')
parser.add_argument('--finetune_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of fintune')
parser.add_argument('--arch', type=str, default=None, #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of fintune')
parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
parser.add_argument('--result_dir', type=str, default='./result', help='results path for saving models and loggers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=180, help='num of training epochs')
# 这个可以去掉
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
# 这个在step学习率下降方式中才有用
parser.add_argument('--lr_decay_step', default='50,100', type=str, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain model path')
parser.add_argument('--ci_dir', type=str, default='', help='ci path')
parser.add_argument('--sparsity', type=str, default=None, help='sparsity of each conv layer')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--adapter_sparsity', type=str, default=None, help='sparsity of each adapter layer')
parser.add_argument('--which', type=str, default=None, help='which dataset')
args = parser.parse_args()

print_freq = (256 * 50) // args.batch_size

# 建立日志
if '2022' in args.result_dir:
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
else:
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 当前时间
    args.result_dir = os.path.join(args.result_dir, now)
logger, writer = utils_append.lgwt_construct(args)
logger.info("args = %s", args)

def adjust_learning_rate(optimizer, epoch, step, len_iter):

    if args.lr_type == 'step':
        # factor = epoch // 30
        if epoch >= 60 and epoch < 90:
            # factor = factor + 1
            lr = args.learning_rate * (0.1 ** 1)
        elif epoch >= 90:
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

    #Warmup
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))

def train(epoch, train_loader, model, criterion, optimizer, scaler = None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()
    #scheduler.step()

    num_iter = len(train_loader)

    print_freq = num_iter // 10

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
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

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main():
    cudnn.benchmark = True
    cudnn.enabled=True

    # 构建模型
    # 根据微调数据集构建模型
    logger.info('==> Building model..')
    # 解析adapter sparsity
    adapter_sparsity = None
    if args.adapter_sparsity:
        adapter_sparsity = utils_append.analysis_sparsity(args.adapter_sparsity)
    elif 'adapter' in args.finetune_arch and 'vgg' not in args.finetune_arch:
        raise ValueError('adapter sparsity is None')

    # 解析sparsity
    if args.sparsity:
        sparsity = utils_append.analysis_sparsity(args.sparsity)
    else:
        raise ValueError('sparsity is None')
    logger.info('sparsity:' + str(sparsity))
    # FINETUNE_CLASSES = utils_append.classes_num(args.finetune_dataset)
    # PRETRAINED_CLASSES = utils_append.classes_num(args.pretrained_dataset)
    # 构建四个模型，一个训练模型，一个计算参数的模型，一个带预训练参数的模型

    if args.graf == False:
        args.arch = args.pretrained_arch

    if 'adapter' in args.finetune_arch:
        # 训练模型使用fintune_arch
        model = eval(args.finetune_arch)(sparsity=sparsity, adapter_sparsity=adapter_sparsity).to(device)
        # 计算参数模型使用finetune_arch
        original_params_model = eval(args.finetune_arch)(sparsity=[0.]*100, adapter_sparsity=[0.]*100).to(device)
        # 加载参数模型使用pretrained_arch
        origin_model = eval(args.pretrained_arch)(sparsity=[0.]*100, adapter_sparsity=[0.0]*100).to(device)
        pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity, adapter_sparsity = adapter_sparsity).to(device)
    else:
        model = eval(args.finetune_arch)(sparsity=sparsity).to(device)
        original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100).to(device)
        origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100).to(device)
        pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity).to(device)

    logger.info(model)

    #计算模型参数量
    # original_params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
    #                                         adapter_sparsity = [0.] * 100).to(device)

    input_size = 224
    logger.info('input size is {}'.format(input_size))
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_params_model,
                                                                       input_size=input_size, dtst=args.finetune_dataset)
    logger.info('model flops is {}, params is {}'.format(flops, params))
    logger.info('model flops reduce ratio is {}, params reduce ratio is {}'.format(flops_ratio, params_ratio))

    # 定义优化器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # criterion_smooth = utils.CrossEntropyLabelSmooth(FINETUNE_CLASSES, args.label_smooth)
    # criterion_smooth = criterion_smooth.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

    start_epoch = 0
    best_top1_acc= 0

    # 加载训练模型，训练模型在不同的数据集上训练
    logger.info('resuming from pretrain model')
    map_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.pretrain_dir, map_location=map_str)

    # 将原始模型参数载入到压缩模型中
    logger.info("载入参数1")
    # utils_append.load_arch_model(args, model, origin_model, ckpt, logger, graf=True)
    logger.info('args graf: {}'.format(args.graf))
    if args.graf == False:
        args.arch = args.pretrained_arch

    # logger.info('random init')
    utils_append.load_arch_model(args, model, origin_model, ckpt, logger, args.graf)

    # train the model
    epoch = start_epoch
    logger.info('device: {}'.format(device))

    # 加载微调数据集
    args.dataset, args.data_dir = args.finetune_dataset, args.finetune_data_dir
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader
    logger.info('the finetune dataset is {}'.format(args.dataset))

    while epoch < args.epochs:
        start = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion, optimizer)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        utils_append.logstore(writer, train_obj, train_top1_acc, valid_obj, valid_top1_acc, epoch)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),}, is_best, args.result_dir)

        epoch += 1
        end = time.time()
        logger.info("=>Best accuracy {:.3f} cost time is {:.3f}".format(best_top1_acc, (end - start)))#

if __name__ == '__main__':
    main()
