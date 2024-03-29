import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import cifar10, imagenet, cifar100, cub
import time
import utils_append
from models.resnet_cifar import resnet_56, resnet_110, resnet_80, resnet_20, resnet_32
from models.resnet_imagenet import resnet_50
from models.adapter_resnet_new import adapter1resnet_56, adapter3resnet_56, adapter5resnet_56, adapter6resnet_56
from models.resnet_tinyimagenet import resnet_tinyimagenet_56
from models.adapter_resnet_tinyimagenet import adapter3resnet_tinyimagenet_56
from models.adapter_resnet_new_new import adapter8resnet_56
from models.vgg_cifar10 import vgg_16_bn
from models.selfsupcon_supcon_adapter_vgg import selfsupcon_supcon_adapter_vgg_16_bn, selfsupcon_supcon_vgg_16_bn
from models.adapter_vgg_cifar10 import adapter_vgg_16_bn, adapter_vgg_16_bn_v4
from models.adapter_resnet_imagenet import adapter15resnet_34
from models.adapter_resnet_new_three import adapter9resnet_56, adapter10resnet_56, adapter11resnet_56, \
    adapter12resnet_56, adapter13resnet_56, adapter14resnet_56, adapter15resnet_56, adapter17resnet_56, \
    adapter16resnet_56, adapter18resnet_56, adapter19resnet_56, adapter20resnet_56, adapter21resnet_56, \
    adapter22resnet_56, adapter23resnet_56, adapter24resnet_56, adapter15resnet_20, adapter19resnet_20, \
    adapter16resnet_32, adapter15resnet_32, adapter25resnet_56
from models.sl_mlp_resnet_cifar import sl_mlp_resnet_56
from models.supcon_adapter_resnet import supcon_adapter15resnet_56
from models.sl_mlp_adapteresnet_cifar import sl_mlp_adapter15resnet_56
from models.selfsupcon_supcon_adapter_resnet import selfsupcon_supcon_adapter15resnet_56, selfsupcon_supcon_resnet_56, \
    selfsupcon_supcon_adapter15resnet_20, selfsupcon_supcon_adapter24resnet_56, selfsupcon_supcon_adapter25resnet_56
'''
运行命令
本地data_dir: /Users/chenjie/dataset/tiny-imagenet-200, 服务器data_dir: /root/autodl-tmp/tiny-imagenet-200
python calculate_feature_maps_n.py --arch adapter3resnet_56 --dataset cifar10 --data_dir ./data \
--pretrain_dir ./pretrained_models/94.86_adapter4resnet_56_cifar10.pth.tar

python calculate_feature_maps_n.py --arch resnet_tinyimagenet_56 --dataset tinyimagenet --data_dir \
/Users/chenjie/dataset/tiny-imagenet-200 \
--pretrain_dir ./pretrained_models/56.19_resnet_tinyimagenet_56_tinyimagenet.pth.tar

python calculate_feature_maps_n.py --arch resnet_56 --dataset cifar100 --data_dir ./data \
--pretrain_dir ./pretrained_models/73.86_resnet_56_cifar100.pth.tar

python calculate_feature_maps_n.py --arch resnet_56 --dataset cifar10 --data_dir ./data \
--pretrain_dir ./pretrained_models/94.78_resnet_56_cifar10.pth.tar

python calculate_feature_maps_n.py --arch adapter3resnet_tinyimagenet_56 --dataset tinyimagenet --data_dir \
/root/autodl-tmp/tiny-imagenet-200 --pretrain_dir \
./pretrained_models/56.2_adapter4resnet_tinyimagenet_56_tinyimagenet.pth.tar

# 计算resnet_80的在cifar10上的feature map
python calculate_feature_maps_n.py --arch resnet_80 --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/94.86_resnet_80_cifar10.pth.tar
# 计算resnet_80在cifar100上的feature_map
python calculate_feature_maps_n.py --arch resnet_80 --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/74.56_resnet_80_cifar100.pth.tar

# 计算adapter5resnet_56的在cifar100上的feature map
python calculate_feature_maps_n.py --arch adapter5resnet_56 --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/74.85_adapter5resnet_56_cifar100.pth.tar

# 计算adapter6resnet_56的在cifar100上的feature map
python calculate_feature_maps_n.py --arch adapter6resnet_56 --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/74.51_adapter6resnet_56_cifar100.pth.tar
'''

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
parser.add_argument('--which', type=str, default=None, help='which dataset')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
# parser.add_argument('--gpu', type=str, default='0', help='gpu id')
args = parser.parse_args()

cudnn.benchmark = True
cudnn.enabled = True
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

# Load pretrained model.
print('Loading Pretrained Model...')
if args.arch == 'adapter15resnet_34':
    model = eval(args.arch)(sparsity=[0.] * 100, adapter_sparsity=[0.] * 100).to(device)
elif args.arch == 'resnet_50':
    model = eval(args.arch)(sparsity=[0.] * 100).to(device)
elif 'adapter' in args.arch:
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
    print('come here')
    print(model)
    checkpoint = torch.load(args.pretrain_dir, map_location=mapstr)
    # model.load_state_dict(torch.load(args.pretrain_dir, map_location=mapstr))
    # exit(1)
    # print(checkpoint)

print('come there')
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
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use 5 batches to get feature maps.
            if batch_idx >= repeat:
               break

            if 'supcon-ce' in args.pretrain_dir or 'supcon' in args.pretrain_dir:
                print('new supcon-ce dataset loader')
                # inputs, targets = inputs[0].to(device), targets.to(device)
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)

            model(inputs)

if args.arch=='vgg_16_bn':

    # if len(args.gpu) > 1:
    #     relucfg = model.module.relucfg
    # else:
    relucfg = model.relucfg
    start = time.time()
    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch=='adapter_vgg_16_bn' or args.arch == 'adapter_vgg_16_bn_v4':

    # if len(args.gpu) > 1:
    #     relucfg = model.module.relucfg
    # else:
    relucfg = model.relucfg
    start = time.time()
    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch == 'selfsupcon_supcon_adapter_vgg_16_bn' or args.arch == 'selfsupcon_supcon_vgg_16_bn':

    # if len(args.gpu) > 1:
    #     relucfg = model.module.relucfg
    # else:
    relucfg = model.relucfg
    start = time.time()
    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch=='resnet_20':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet18 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(3):
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

elif args.arch=='resnet_32':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet32 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(5):
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

elif args.arch=='resnet_56':

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

elif args.arch=='sl_mlp_resnet_56':

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

elif args.arch=='resnet_80':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(13):
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

elif args.arch=='resnet_tinyimagenet_56':

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

elif args.arch == 'adapter1resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # adapter1resnet56 per block
    # adapter结构的输出也进行修改
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].adapter.relu
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].adapter.convup
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter3resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # adapter1resnet56 per block
    # adapter结构的输出也进行修改
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].adapter.relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].adapter.relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter5resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # adapter1resnet56 per block
    # adapter结构的输出也进行修改
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

            cov_layer = block[j].adapter.relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].adapter.relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter6resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # adapter1resnet56 per block
    # adapter结构的输出也进行修改
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].adapter.relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].adapter.relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter8resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有13层
        # 其中2， 5， 8， 11为adapter结构
        for j in range(13):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter9resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter10resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter11resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter12resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter13resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter14resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter15resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter15resnet_20' or args.arch == 'adapter19resnet_20':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(3):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter15resnet_32' :

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有5层
        for j in range(5):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'resnet_50':
    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    i = 0
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(3):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu3
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 1
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(4):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu3
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 2
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(6):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu3
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 3
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(3):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu3
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

elif args.arch == 'adapter15resnet_34':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    i = 0
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(3):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 1
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(4):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 2
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(6):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

    i = 3
    block = eval('model.layer%d' % (i + 1))
    # 每一个stage有5层
    for j in range(3):
        cov_layer = block[j].relu1
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

        cov_layer = block[j].relu2
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()
        cnt += 1

elif args.arch == 'adapter16resnet_32':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有5层
        for j in range(5):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'selfsupcon_supcon_adapter15resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'selfsupcon_supcon_adapter24resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'selfsupcon_supcon_adapter25resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'selfsupcon_supcon_adapter15resnet_20':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(3):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'selfsupcon_supcon_resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'sl_mlp_adapter15resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'supcon_adapter15resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter17resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter18resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter19resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter20resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter21resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter22resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter23resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter24resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        # 每一个stage有9层
        # 其中第8层为adapter结构
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            # cov_layer = block[j].adapter.relu1
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1
            #
            # cov_layer = block[j].adapter.relu2
            # handler = cov_layer.register_forward_hook(get_feature_hook)
            # inference()
            # handler.remove()
            # cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch == 'adapter3resnet_tinyimagenet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # adapter1resnet56 per block
    # adapter结构的输出也进行修改
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].adapter.relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].adapter.relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1
elif args.arch=='resnet_110':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch=='resnet_50':
    cov_layer = eval('model.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet50 per bottleneck
    for i in range(4):
        block = eval('model.layer%d' % (i + 1))
        for j in range(model.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            if j==0:
                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
