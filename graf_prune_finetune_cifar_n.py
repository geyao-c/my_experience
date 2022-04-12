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
    adapter20resnet_56, adapter21resnet_56, adapter22resnet_56
import utils_append
import utils
import time

'''
嫁接裁剪微调，在数据集A上裁剪，然后在数据集B上微调
更换模型进行裁剪时一般需要修改的参数为 --dataset, --result_dir, --arch, --ci_dir, --pretrain_dir --sparsity
# 嫁接裁剪cifar10 to cifar100 
# 裁剪48%
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir \
./data --pretrained_arch resnet_80 --finetune_arch resnet_80 \
--result_dir ./result/graf_pruned/94.86cifar10tocifar100_resnet_80_pruned_48_2 \
--ci_dir ./calculated_ci/94.86_resnet_80_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.86_resnet_80_cifar10.pth.tar --sparsity [0.]+[0.15]*2+[0.4]*13+[0.4]*13+[0.4]*13

# 裁剪70%
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir \
./data --pretrained_arch resnet_80 --finetune_arch resnet_80 \
--result_dir ./result/graf_pruned/94.86cifar10tocifar100_resnet_80_pruned_71_2 \
--ci_dir ./calculated_ci/94.86_resnet_80_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.86_resnet_80_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*13+[0.6]*13+[0.75]*13

resnet_80裁剪48%的参数裁剪率设置
sparsity = '[0.]+[0.15]*2+[0.4]*13+[0.4]*13+[0.4]*13'
resnet_80裁剪71%的参数裁剪率设置
sparsity = '[0.]+[0.4]*2+[0.5]*13+[0.6]*13+[0.75]*13'
# ------------------------------------------------------------------------------------------------------------------

本地data_dir: /Users/chenjie/dataset/tiny-imagenet-200, 服务器data_dir: /root/autodl-tmp/tiny-imagenet-200
# graf adapter4resnet to adapter4resnet_tinyimagenet
# 裁剪48%
--sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9
--adapter_sparsity [0.]*9+[0.1]*9+[0.2]*9
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset tinyimagenet --finetune_data_dir \
/root/autodl-tmp/tiny-imagenet-200 --pretrained_arch adapter3resnet_56 --finetune_arch adapter3resnet_tinyimagenet_56 \
--result_dir ./result/graf_pruned/94.86cifar10totinyimagenet_adapter3resnet_tinyimagenet_56_pruned_48_1 \
--ci_dir ./calculated_ci/94.86_adapter4resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.86_adapter4resnet_56_cifar10.pth.tar --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 \
--adapter_sparsity [0.]*9+[0.1]*9+[0.2]*9
# 裁剪71%
--sparsity = [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
--adapter_sparsity [0.2]*9+[0.3]*9+[0.35]*9
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset tinyimagenet --finetune_data_dir \
/root/autodl-tmp/tiny-imagenet-200 --pretrained_arch adapter3resnet_56 --finetune_arch adapter3resnet_tinyimagenet_56 \
--result_dir ./result/graf_pruned/94.86cifar10totinyimagenet_adapter4resnet_tinyimagenet_56_pruned_71_1 \
--ci_dir ./calculated_ci/94.86_adapter4resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.86_adapter4resnet_56_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 \
--adapter_sparsity [0.2]*9+[0.3]*9+[0.35]*9


# resnet56 to resnet_tinyimagenet_56裁剪
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset tinyimagenet --finetune_data_dir /root/autodl-tmp/tiny-imagenet-200 \
--pretrained_arch resnet_56 --finetune_arch resnet_tinyimagenet_56 \
--result_dir ./result/graf_pruned/94.78cifar10totinyimagenet_resnet_tinyimagenet_56_pruned_71_1 \
--ci_dir ./calculated_ci/94.78_resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.78_resnet_56_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
--sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
[0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9

# 裁剪48%命令行
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir ./data \
--result_dir ./result/graf_pruned/94.56cifar10tocifar100_adapter4resnet_56_pruned_48_1 \
--arch adapter3resnet_56 --ci_dir ./calculated_ci/94.56_adapter4resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.56_adapter4resnet_56_cifar10.pth.tar --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9 \
--adapter_sparsity [0.]*9+[0.2]*9+[0.2]*9
# 裁剪48%resnet56
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir ./data \
--result_dir ./result/graf_pruned/94.78cifar10tocifar100_resnet_56_pruned_71_2 \
--arch resnet_56 --ci_dir ./calculated_ci/94.78_resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.78_resnet_56_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
# 裁剪resnet不需要adapter_sparsity参数
--adapter_sparsity [0.]*9+[0.2]*9+[0.2]*9

# adapter裁剪命令
裁剪48%
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir ./data \
--result_dir ./result/graf_pruned/94.86cifar10tocifar100_adapter4resnet_56_pruned_71_3 \
--arch adapter3resnet_56 --ci_dir ./calculated_ci/94.86_adapter4resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.86_adapter4resnet_56_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 \
--adapter_sparsity [0.2]*9+[0.4]*9+[0.5]*9

# 裁剪71%命令行
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir ./data \
--result_dir ./result/graf_pruned/94.56cifar10tocifar100_adapter4resnet_56_pruned_71_2 \
--arch adapter3resnet_56 --ci_dir ./calculated_ci/94.56_adapter4resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/94.56_adapter4resnet_56_cifar10.pth.tar --sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 \
--adapter_sparsity [0.15]*9+[0.45]*9+[0.55]*9

# sparsity第一项为第一层的裁剪率，第二项和第三项为第一个和第二个stage的输出裁剪率，
# 最后三项为三个stage中间层的裁剪率
# sparsity = [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9  裁剪48%
# sparsity = [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9   裁剪71%
# mid_sparsity = [0.]*9+[0.2]*9+[0.2]*9 裁剪48%
# adapter_sparsity = [0.1]*9+[0.3]*9+[0.3]*9 裁剪71%

已经试了这个: [0.2]*9+[0.4]*9+[0.6]*9
# 再试一下这个
# [0.15]*9+[0.45]*9+[0.55]*9
# [0.2]*9+[0.4]*9+[0.5]*9
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先解析参数
parser = argparse.ArgumentParser("CIFAR prune training")
# 训练数据集
parser.add_argument('--pretrained_dataset', type=str, default='cifar10', help='model train dataset used')
# 微调数据集
parser.add_argument('--finetune_dataset', type=str, default='cifar100', help='model train dataset used')
parser.add_argument('--dataset', type=str, default=None, help='model train dataset used')
# 微调数据集文件路径
parser.add_argument('--finetune_data_dir', type=str, default='./data', help='path to dataset')
parser.add_argument('--data_dir', type=str, default=None, help='path to dataset')
# parser.add_argument('--arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
#                     help='architecture to calculate feature maps')
parser.add_argument('--pretrained_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture of pretrained')
parser.add_argument('--finetune_arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
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
args = parser.parse_args()

def main():
    cudnn.benchmark = True
    cudnn.enabled=True

    print_freq = (256 * 50) // args.batch_size

    # 构建日志和writer
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
    # 构建三个模型，一个训练模型，一个计算参数的模型，一个带预训练参数的模型
    if 'adapter' in args.finetune_arch:
        # 训练模型使用fintune_arch
        model = eval(args.finetune_arch)(sparsity=sparsity, num_classes=FINETUNE_CLASSES, adapter_sparsity = adapter_sparsity).to(device)
        # 计算参数模型使用finetune_arch
        original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
                                                adapter_sparsity=[0.] * 100).to(device)
        # 加载参数模型使用pretrained_arch
        origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES, adapter_sparsity=[0.0] * 100).to(device)
    else:
        model = eval(args.finetune_arch)(sparsity=sparsity, num_classes=FINETUNE_CLASSES).to(device)
        original_params_model = eval(args.finetune_arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES).to(device)
        origin_model = eval(args.pretrained_arch)(sparsity=[0.] * 100, num_classes=PRETRAINED_CLASSES).to(device)
    logger.info(model)

    #计算模型参数量
    # original_params_model = eval(args.arch)(sparsity=[0.] * 100, num_classes=FINETUNE_CLASSES,
    #                                         adapter_sparsity = [0.] * 100).to(device)
    if 'tinyimagenet' in args.finetune_dataset:
        input_size = 64
    else:
        input_size = 32
    flops, params, flops_ratio, params_ratio = utils_append.cal_params(model, device, original_params_model, input_size=input_size)
    logger.info('model flops is {}, params is {}'.format(flops, params))
    logger.info('model flops reduce ratio is {}, params reduce ratio is {}'.format(flops_ratio, params_ratio))

    # 定义优化器
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
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
    utils_append.load_arch_model(args, model, origin_model, ckpt, logger, graf=True)

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
