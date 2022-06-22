import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import datetime
import os
from models.resnet_cifar import resnet_56,resnet_110, resnet_80
from models.adapter_resnet_new import adapter1resnet_56, adapter3resnet_56, \
    adapter5resnet_56, adapter6resnet_56
from models.resnet_tinyimagenet import resnet_tinyimagenet_56
from models.adapter_resnet_tinyimagenet import adapter3resnet_tinyimagenet_56
from models.adapter_resnet_new_new import adapter8resnet_56
from models.adapter_resnet_new_three import adapter9resnet_56, adapter10resnet_56, \
    adapter11resnet_56, adapter12resnet_56, adapter13resnet_56, adapter14resnet_56, \
    adapter15resnet_56, adapter17resnet_56, adapter18resnet_56, adapter19resnet_56, \
    adapter20resnet_56, adapter21resnet_56, adapter22resnet_56, adapter23resnet_56, \
    adapter24resnet_56
from models.supcon_adapter_resnet import supcon_adapter15resnet_56
from models.sl_mlp_resnet_cifar import sl_mlp_resnet_56
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
import utils_append
import utils
import time

'''
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先解析参数
parser = argparse.ArgumentParser("CIFAR prune training")
# 训练数据集
parser.add_argument('--pretrained_dataset', type=str, default='cifar10', help='model train dataset used')
# 微调数据集
parser.add_argument('--finetune_dataset', type=str, default='cifar100', help='model train dataset used')
parser.add_argument('--dataset', type=str, default=None, help='model train dataset used')
parser.add_argument('--graf', action="store_false", help='graf pruned or not')
# 微调数据集文件路径
parser.add_argument('--finetune_data_dir', type=str, default='./data', help='path to dataset')
parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
# parser.add_argument('--arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
#                     help='architecture to calculate feature maps')
parser.add_argument('--pretrained_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of pretrained')
parser.add_argument('--finetune_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of fintune')
parser.add_argument('--arch', type=str, default=None, #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of fintune')
parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
parser.add_argument('--result_dir', type=str, default='./result', help='results path for saving models and loggers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
# 这个可以去掉
parser.add_argument('--label_smooth', type=float, default=0, help='label smoothing')
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
parser.add_argument('--split', type=str, default='1', help='batch size')
# cutmix参数
parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float, help='cutmix probability')
# mixup参数
parser.add_argument('--mixup_alpha', default=0, type=float,
                    help='mixup interpolation coefficient (default: 0)')
# cutout参数
parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, help='length of the holes')
args = parser.parse_args()

def main():
    cudnn.benchmark = True
    cudnn.enabled=True

    print_freq = (256 * 50) // args.batch_size

    # 建立日志
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  # 当前时间
    args.result_dir = os.path.join(args.result_dir, now)
    logger, writer = utils_append.lgwt_construct(args)
    logger.info("args = %s", args)

    # 加载微调数据集
    args.dataset, args.data_dir = args.finetune_dataset, args.finetune_data_dir
    train_loader, val_loader = utils_append.dstget(args)
    logger.info('the finetune dataset is {}'.format(args.dataset))

    # 构建模型
    # 根据微调数据集构建模型
    logger.info('==> Building model..')
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

    logger.info('sparsity:' + str(sparsity))
    FINETUNE_CLASSES = utils_append.classes_num(args.finetune_dataset)
    PRETRAINED_CLASSES = utils_append.classes_num(args.pretrained_dataset)
    # 构建四个模型，一个训练模型，一个计算参数的模型，一个带预训练参数的模型
    if 'adapter' in args.finetune_arch:
        # 训练模型使用fintune_arch
        model = eval(args.finetune_arch)(sparsity=sparsity, num_classes=FINETUNE_CLASSES, adapter_sparsity = adapter_sparsity,
                                         dataset=args.finetune_dataset).to(device)
        # 计算参数模型使用finetune_arch
        original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
                                                adapter_sparsity=[0.] * 100, dataset=args.finetune_dataset).to(device)
        # 加载参数模型使用pretrained_arch
        origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES, adapter_sparsity=[0.0] * 100,
                                                  dataset=args.pretrained_dataset).to(device)
        pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity, num_classes=PRETRAINED_CLASSES, adapter_sparsity = adapter_sparsity,
                                                         dataset=args.pretrained_dataset).to(device)
    else:
        model = eval(args.finetune_arch)(sparsity=sparsity, num_classes=FINETUNE_CLASSES, dataset=args.finetune_dataset).to(device)
        original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
                                                         dataset=args.finetune_dataset).to(device)
        origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES,
                                                  dataset=args.pretrained_dataset).to(device)
        pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity, num_classes=PRETRAINED_CLASSES,
                                                         dataset=args.pretrained_dataset).to(device)

    logger.info(model)

    #计算模型参数量
    # original_params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
    #                                         adapter_sparsity = [0.] * 100).to(device)
    if 'tinyimagenet' in args.finetune_dataset:
        input_size = 64
    elif 'dtd' == args.finetune_dataset:
        input_size = 128
    else:
        input_size = 32
    logger.info('input size is {}'.format(input_size))
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_params_model, input_size=input_size)
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
    # PRETRAINED_CLASSES = utils_append.classes_num(args.pretrained_dataset)
    # origin_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES, adapter_sparsity=[0.0]*100).to(device)
    map_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.pretrain_dir, map_location=map_str)

    # 将原始模型参数载入到压缩模型中
    logger.info("载入参数1")
    # utils_append.load_arch_model(args, model, origin_model, ckpt, logger, graf=True)
    logger.info('args graf: {}'.format(args.graf))
    if args.graf == False:
        args.arch = args.pretrained_arch
    utils_append.load_arch_model(args, model, origin_model, ckpt, logger, args.graf)

    # 压缩原始模型，得到压缩后的精度
    # logger.info("载入参数2")
    # logger.info('pruned origin model: ', pruned_origin_model)
    # utils_append.overall_load_arch_model(args, pruned_origin_model, origin_model, ckpt, logger, graf=True)
    # 加载训练前的数据集
    # args.dataset = args.pretrained_dataset
    # if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    #     args.data_dir = "./data"

    # print('args dataset: ', args.dataset)
    # pretrained_train_loader, pretrained_val_loader = utils_append.dstget(args)
    # logger.info(pruned_origin_model.state_dict().keys())
    # 计算压缩后的精度
    # pruned_valid_obj, pruned_valid_top1_acc, pruned_valid_top5_acc = utils_append.validate(None, pretrained_val_loader, pruned_origin_model,
    #                                                                   criterion, args, logger, device)
    # logger.info("=>after pruned accuracy is {:.3f}".format(pruned_valid_top1_acc))

    # train the model
    epoch = start_epoch
    if args.beta > 0 and args.cutmix_prob > 0:
        logger.info("use cut mix")
    if args.cutout:
        logger.info("use cutout")
    if args.mixup_alpha > 0:
        logger.info("use mixup")
    logger.info('device: {}'.format(device))

    while epoch < args.epochs:
        start = time.time()
        train_obj, train_top1_acc,  train_top5_acc = utils_append.train(epoch,  train_loader, model, criterion,
                                                                        optimizer, args, logger, print_freq, device)#, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = utils_append.validate(epoch, val_loader, model, criterion, args, logger, device)
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
    print('hello world')
    main()
