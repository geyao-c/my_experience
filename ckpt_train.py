import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models.resnet_cifar import resnet_56,resnet_110, resnet_80
from models.adapter_resnet_new import adapter1resnet_56, adapter3resnet_56, \
    adapter5resnet_56, adapter6resnet_56
from models.resnet_tinyimagenet import resnet_tinyimagenet_56
from models.adapter_resnet_tinyimagenet import adapter3resnet_tinyimagenet_56
from models.adapter_resnet_new_new import adapter8resnet_56
from models.adapter_resnet_new_three import adapter9resnet_56, adapter10resnet_56, \
    adapter11resnet_56, adapter12resnet_56, adapter13resnet_56, adapter14resnet_56, \
    adapter15resnet_56, adapter17resnet_56, adapter18resnet_56, adapter19resnet_56, \
    adapter20resnet_56, adapter21resnet_56, adapter22resnet_56, adapter23resnet_56
import utils_append
import utils
import time

'''
python ckpt_train.py --dataset svhn --data_dir ./data/svhn --arch adapter22resnet_56 \
--result_dir ./result/ckpt_train/94.43cifar10tosvhn_adapter22resnet_56_pruned_48 \
 --batch_size 128 --epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/model_best1.pth.tar --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 --adapter_sparsity [0.4]+[0.415]+[0.417]
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先解析参数
parser = argparse.ArgumentParser("ckpt prune training")
# 数据集参数
parser.add_argument('--dataset', type=str, default='cifar10', help='model train dataset used')
parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# 网络结构参数
parser.add_argument('--arch', type=str, default=None, #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of fintune')
parser.add_argument('--sparsity', type=str, default=None, help='sparsity of each conv layer')
parser.add_argument('--adapter_sparsity', type=str, default=None, help='sparsity of each adapter layer')

# 日志存储参数
parser.add_argument('--result_dir', type=str, default='./result', help='results path for saving models and loggers')
# 学习率参数
parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
# ckpt路径参数
parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain model path')

# 这个可以去掉
# parser.add_argument('--label_smooth', type=float, default=0, help='label smoothing')
# 这个在step学习率下降方式中才有用
# parser.add_argument('--lr_decay_step', default='50,100', type=str, help='learning rate')
# parser.add_argument('--ci_dir', type=str, default='', help='ci path')
# parser.add_argument('--gpu', type=str, default='0', help='gpu id')
args = parser.parse_args()

def main():
    cudnn.benchmark = True
    cudnn.enabled=True

    print_freq = (256 * 50) // args.batch_size

    # 构建日志和writer
    logger, writer = utils_append.lgwt_construct(args)
    logger.info("args = %s", args)

    # args.dataset, args.data_dir = args.finetune_dataset, args.finetune_data_dir
    # 直接加载数据集
    train_loader, val_loader = utils_append.dstget(args)
    logger.info('the train dataset is {}'.format(args.dataset))

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
    CLASSES = utils_append.classes_num(args.dataset)
    # 构建四个模型，一个训练模型，一个计算参数的模型，一个带预训练参数的模型
    if 'adapter' in args.arch:
        # 训练模型使用fintune_arch
        model = eval(args.arch)(sparsity=sparsity, num_classes=CLASSES, adapter_sparsity = adapter_sparsity).to(device)
        # 计算参数模型使用finetune_arch
        # original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
        #                                         adapter_sparsity=[0.] * 100).to(device)
        # 加载参数模型使用pretrained_arch
        # origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES, adapter_sparsity=[0.0] * 100).to(device)
        # pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity, num_classes=PRETRAINED_CLASSES, adapter_sparsity = adapter_sparsity).to(device)
    else:
        model = eval(args.finetune_arch)(sparsity=sparsity, num_classes=CLASSES).to(device)
        # original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES).to(device)
        # origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES).to(device)
        # pruned_origin_model = eval(args.pretrained_arch)(sparsity=sparsity, num_classes=PRETRAINED_CLASSES).to(device)

    logger.info(model)

    #计算模型参数量
    # original_params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
    #                                         adapter_sparsity = [0.] * 100).to(device)
    # if 'tinyimagenet' in args.finetune_dataset:
    #     input_size = 64
    # else:
    #     input_size = 32
    # flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_params_model, input_size=input_size)
    # logger.info('model flops is {}, params is {}'.format(flops, params))
    # logger.info('model flops reduce ratio is {}, params reduce ratio is {}'.format(flops_ratio, params_ratio))

    # 定义优化器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    # criterion_smooth = criterion_smooth.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

    # 从ckpt path中读取待训练模型和训练相关参数
    logger.info('resuming from pretrain model')
    map_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.pretrain_dir, map_location=map_str)
    # state_dict = ckpt['state_dict']
    # logger.info('state dicts: {}'.format(state_dict.keys()))
    model_state_dict = model.state_dict()
    ckpt_state_dict = ckpt['state_dict']
    for key in model_state_dict.keys():
        logger.info('key: {}'.format(key))
        model_state_dict[key] = ckpt_state_dict[key]
    # model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt['epoch']
    best_top1_acc= ckpt['best_top1_acc']
    optimizer.load_state_dict(ckpt['optimizer'])

    # train the model
    epoch = start_epoch
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
  main()
