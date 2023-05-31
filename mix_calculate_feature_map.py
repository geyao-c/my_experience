import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import cifar10, imagenet, cifar100, cub, svhn
import time
import utils_append
from models.resnet_cifar import resnet_56, resnet_110, resnet_80, resnet_20, resnet_32

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='Calculate Feature Maps')
parser.add_argument('--arch', type=str, default='resnet_56', #choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture to calculate feature maps')
parser.add_argument('--dataset', type=str, default='cifar10', # choices=('cifar10','imagenet', 'cifar100', 'cub'),
                    help='dataset used')
parser.add_argument('--data_dir', type=str, default='./data', help='dataset path')
parser.add_argument('--pretrain_dir', type=str, default=None, help='dir for the pretriained model to calculate feature maps')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for one batch.')
parser.add_argument('--repeat', type=int, default=5, help='the number of different batches for calculating feature maps.')
parser.add_argument('--save_dir', type=str, default=None, help='save dir')
# parser.add_argument('--gpu', type=str, default='0', help='gpu id')
args = parser.parse_args()

cudnn.benchmark = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train_loader1, _ = cifar10.load_cifar_data(args)
train_loader2, _ = svhn.load_svhn_data(args)
CLASSES = 20

# Load pretrained model.
print('Loading Pretrained Model...')
if 'adapter' in args.arch:
    model = eval(args.arch)(sparsity=[0.]*100, num_classes=CLASSES, adapter_sparsity=[0.]*100,
                            dataset=args.dataset).to(device)
else:
    model = eval(args.arch)(sparsity=[0.]*100, num_classes=CLASSES, dataset=args.dataset).to(device)
# print(model)
mapstr = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(args.arch)
if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
    print('here')
    checkpoint = torch.load(args.pretrain_dir, map_location=mapstr)
else:
    checkpoint = torch.load(args.pretrain_dir, map_location=mapstr)

if args.arch=='resnet_50':
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint['state_dict'], strict=False)

conv_index = torch.tensor(1)

# 提取出模型准确率
if args.save_dir is None:
    model_accu = str(args.pretrain_dir.split('/')[2].split('_')[0])
    print('model accuracy: {}'.format(model_accu))

def get_feature_hook(self, input, output):
    global conv_index

    if args.save_dir is None:
        dirpath = os.path.join('../conv_feature_map', model_accu + '_' + args.arch + '_' + args.dataset + '_repeat%d' % (args.repeat))
    else:
        dirpath = args.save_dir
    # dirpath = os.path.join('conv_feature_map', args.arch + '_repeat%d' % (args.repeat))
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    filepath = os.path.join(dirpath, 'conv_feature_map_' + str(conv_index) + '.npy')
    np.save(filepath, output.cpu().numpy())
    print('{} has been saved'.format(filepath))
    conv_index += 1

def inference():
    model.eval()
    repeat = args.repeat
    iter_train_loader1 = iter(train_loader1)
    iter_train_loader2 = iter(train_loader2)
    count = 0
    with torch.no_grad():
        while count < repeat:
            if count % 2 == 0:
                inputs, targets = next(iter_train_loader1)
            else:
                inputs, targets = next(iter_train_loader2)
            count += 1
            inputs, targets = inputs.to(device), targets.to(device)
            model(inputs)

if args.arch=='resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1
