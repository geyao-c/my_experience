import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

from models.efficientnet import efficientnet_b0, efficientnet_b0_changed_v2, efficientnet_b0_changed_v4
from models.mobilenetv2 import mobilenet_v2
import utils_append

from data import imagenet
# import utils.common as utils

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument('--data_dir', type=str, default='./data', help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','imagenet'), help='dataset')
parser.add_argument('--save_dir', type=str, default=None, help='save dir')
# parser.add_argument('--job_dir', type=str, default='result/tmp', help='The directory where the summaries will be stored.')
parser.add_argument('--arch', type=str, default='vgg_16_bn', help='The architecture to prune')
parser.add_argument('--pretrain_dir', type=str, default=None, help='load the model from the specified checkpoint')
parser.add_argument('--repeat', type=int, default=5, help='The num of batch to get rank.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
parser.add_argument('--gpu', type=str, default='0', help='Select gpu to use')
args = parser.parse_args()

"""
python rank_generation.py --data_dir ./data --dataset cifar10 --arch mobilenet_v2 --pretrain_dir ./pretrained_models/92.38_mobilenet_v2_cifar10.pth.tar
"""

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
if 'supcon-ce' in args.pretrain_dir:
    print('loader dataset from supcon dataset')
    train_loader, _ = utils_append.supcon_dstget(args)
elif 'selfsupcon-supcon' in args.pretrain_dir or 'selfsupcon-supcon' in args.pretrain_dir:
    print('loader dataset from supcon dataset')
    print('new selfsupcon-supcon in args pretrain dir')
    train_loader, _ = utils_append.supcon_dstget(args)
    # train_loader, _ = utils_append.dstget(args)
else:
    print('loader dataset from normal dataset')
    train_loader, _ = utils_append.dstget(args)
CLASSES = utils_append.classes_num(args.dataset)

# 加载模型
print('Loading Pretrained Model...')
if 'adapter' in args.arch:
    model = eval(args.arch)(sparsity=[0.]*100, num_classes=CLASSES, adapter_sparsity=[0.]*100,
                            dataset=args.dataset).to(device)
else:
    model = eval(args.arch)(sparsity=[0.]*100, num_classes=CLASSES, dataset=args.dataset).to(device)
print(model)
mapstr = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(args.arch)
if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
    checkpoint = torch.load(args.pretrain_dir, map_location=mapstr)
else:
    checkpoint = torch.load(args.pretrain_dir, map_location=mapstr)

if args.arch=='resnet_50':
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint['state_dict'], strict=False)

conv_index = torch.tensor(1)

criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)

# 提取出模型准确率
model_accu = None
if args.save_dir is None:
    model_accu = str(args.pretrain_dir.split('/')[2].split('_')[0])
    print('model accuracy: {}'.format(model_accu))

dirpath = None
if args.save_dir is None:
    dirpath = os.path.join('./rank_conv', model_accu + '_' + args.arch + '_' + args.dataset + '_repeat%d' % (args.repeat))
else:
    dirpath = args.save_dir
# dirpath = os.path.join('conv_feature_map', args.arch + '_repeat%d' % (args.repeat))
if not os.path.isdir(dirpath):
    os.makedirs(dirpath)

#get feature map of certain layer via hook
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def inference():
    global best_acc
    model.eval()
    limit = args.repeat

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # use the first 5 batches to estimate the rank.
            if batch_idx >= limit:
               break

            if 'supcon-ce' in args.pretrain_dir or 'supcon' in args.pretrain_dir:
                print('new supcon-ce dataset loader')
                inputs, targets = inputs[0].to(device), targets.to(device)
                # inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)

            model(inputs)

if args.arch=='mobilenet_v2':
    # global feature_result
    # global total

    cov_layer = eval('model.features[0]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
    np.save(filepath, feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    conv_index += 1

    cnt = 1
    for i in range(1, 19):
        if i == 1:
            block = eval('model.features[%d].conv' % (i))
            relu_list = [2, 4]
        elif i == 18:
            block = eval('model.features[%d]' % (i))
            relu_list = [2]
        else:
            block = eval('model.features[%d].conv' % (i))
            relu_list = [2, 5, 7]

        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
            np.save(filepath, feature_result.numpy())
            conv_index += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            print('{} has been saved'.format(conv_index))

elif args.arch == 'efficientnet_b0_changed_v2':
    # 第一层
    cov_layer = eval('model.stem_conv')
    handler = cov_layer[2].sigmoid.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
    np.save(filepath, feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    conv_index += 1

    for i in range(0, 16):
        block = eval('model.blocks[%d].conv' % (i))
        if i == 0:
            t_list = [(0, 2), (1, 'conv', 1), (1, 'conv', 3), (2, 1)]
        else:
            t_list = [(0, 2), (1, 2), (2, 'conv', 1), (2, 'conv', 3), (3, 1)]
        for item in t_list:
            if len(item) == 2:
                cov_layer = block[item[0]][item[1]]
            elif len(item) == 3:
                cov_layer = eval('model.blocks[%d].conv[%d].conv[%d]' % (i, item[0], item[2]))
                # cov_layer = block[item[0]].item[1][item[2]]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
            np.save(filepath, feature_result.numpy())
            conv_index += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            print('{} has been saved'.format(conv_index))

    # 最后一层
    cov_layer = eval('model.head_conv')
    handler = cov_layer[2].sigmoid.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
    np.save(filepath, feature_result.numpy())
    conv_index += 1
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

elif args.arch == 'efficientnet_b0_changed_v4':
    # 第一层
    cov_layer = eval('model.stem_conv')
    handler = cov_layer[2].sigmoid.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
    np.save(filepath, feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    conv_index += 1

    for i in range(0, 16):
        block = eval('model.blocks[%d].conv' % (i))
        if i == 0:
            t_list = [(0, 2), (1, 'conv', 1), (1, 'conv', 3), (2, 1)]
        else:
            t_list = [(0, 2), (1, 2), (2, 'conv', 1), (2, 'conv', 3), (3, 1)]
        for item in t_list:
            if len(item) == 2:
                cov_layer = block[item[0]][item[1]]
            elif len(item) == 3:
                cov_layer = eval('model.blocks[%d].conv[%d].conv[%d]' % (i, item[0], item[2]))
                # cov_layer = block[item[0]].item[1][item[2]]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
            np.save(filepath, feature_result.numpy())
            conv_index += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            print('{} has been saved'.format(conv_index))

    # 最后一层
    cov_layer = eval('model.head_conv')
    handler = cov_layer[2].sigmoid.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    filepath = os.path.join(dirpath, 'rank_conv_' + str(conv_index) + '.npy')
    np.save(filepath, feature_result.numpy())
    conv_index += 1
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
