import utils
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import re
from data import cifar10, cifar100, cub, tinyimagenet, svhn, dtd, mnist
import torch
from thop import profile
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import math
import time
from data import supconcifar10, supconcifar100

# 构建logger和writer
def lgwt_construct(args):
    # 建立日志
    utils.record_config(args)  # 建立config.txt文件
    logger = utils.get_logger(os.path.join(args.result_dir, 'logger' + '.log'))  # 运行时日志文件
    writer = SummaryWriter(os.path.join(args.result_dir, 'tensorboard'))  # tensorboard文件夹
    # 返回日志和tensorboard写入器
    return logger, writer

"""
解析sparsity
# sparsity第一项为第一层的裁剪率，第二项和第三项为第一个和第二个stage的输出裁剪率，
# 最后三项为三个stage中间层的裁剪率
# sparsity = [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9  裁剪48%
# sparsity = [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9   裁剪71%
"""
def analysis_sparsity(sparsity):
    cprate_str = sparsity
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        # print(find_cprate)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    sparsity = cprate
    # 返回解析完成的sparsity数组
    return sparsity

# 根据args.dataset, 返回对应的数据集
def dstget(args):
    dsetlist = ['cifar10', 'cifar100', 'cub', 'tinyimagenet', 'svhn', 'dtd', 'mnist']
    dldfunlist = [cifar10.load_cifar_data, cifar100.load_cifar_data, cub.load_cub_data, tinyimagenet.load_tinyimagenet_data,
                  svhn.load_svhn_data, dtd.load_dtd_data, mnist.load_mnist_data]
    idx = dsetlist.index(args.dataset)
    train_loader, val_loader = dldfunlist[idx](args)
    return train_loader, val_loader

def supcon_dstget(args):
    dsetlist = ['cifar10', 'cifar100', 'cub', 'tinyimagenet', 'svhn', 'dtd']
    dldfunlist = [supconcifar10.load_cifar_data, supconcifar100.load_cifar_data]
    idx = dsetlist.index(args.dataset)
    train_loader, val_loader = dldfunlist[idx](args)
    return train_loader, val_loader

# 根据args，返回最后一层的神经元数量
def classes_num(datasetname):
    dsetlist = ['cifar10', 'cifar100', 'cub', 'tinyimagenet', 'svhn', 'dtd', 'mnist']
    # 最后一层全连接层神经元数量
    clslist = [10, 100, 200, 200, 10, 47, 10]
    idx = dsetlist.index(datasetname)
    CLASSES = clslist[idx]
    return CLASSES

# 计算模型参数量
def cal_params(model, device, original_model=None, input_size=32, dtst=None):
    if dtst == 'mnist':
        input_image_size = 28
        input_image = torch.randn(1, 1, input_image_size, input_image_size).to(device)
    else:
        input_image_size = 32
        input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    flops, params = profile(model, inputs=(input_image,))
    flops_ratio, params_ratio = None, None
    if original_model is not None:
        original_flops, original_params = profile(original_model, inputs=(input_image,))
        flops_ratio, params_ratio = 1 - flops / original_flops, 1 - params / original_params
    return flops, params, flops_ratio, params_ratio

def load_vgg_model(args, model, oristate_dict, logger, name_base=''):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                ci = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(ci)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def graf_load_vgg_model(args, model, oristate_dict, logger, name_base=''):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                ci = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(ci)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def overall_load_resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        80: [13, 13, 13],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []
    all_bn_weight = []

    prefix = args.ci_dir + '/ci_conv'
    subfix = ".npy"

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base + conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                bn_name = layer_name + str(k) + '.bn' + str(l + 1)
                bn_weight_name = bn_name + '.weight'
                bn_bias_name = bn_name + '.bias'
                bn_runing_mean_name = bn_name + '.running_mean'
                bn_running_var_name = bn_name + '.running_var'
                bn_num_batches_tracked_name = bn_name + '.num_batches_tracked'
                all_bn_weight.extend([bn_name])

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                    # 加载bn层
                    for index_i, i in enumerate(select_index):
                        state_dict[name_base + bn_weight_name][index_i] = oristate_dict[bn_weight_name][i]
                        state_dict[name_base + bn_bias_name][index_i] = oristate_dict[bn_bias_name][i]
                        state_dict[name_base + bn_runing_mean_name][index_i] = oristate_dict[bn_runing_mean_name][i]
                        state_dict[name_base + bn_running_var_name][index_i] = oristate_dict[bn_running_var_name][i]
                        state_dict[name_base + bn_num_batches_tracked_name] = oristate_dict[bn_num_batches_tracked_name]


                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None
                    # 加载bn层
                    state_dict[name_base + bn_weight_name] = oristate_dict[bn_weight_name]
                    state_dict[name_base + bn_bias_name] = oristate_dict[bn_bias_name]
                    state_dict[name_base + bn_runing_mean_name] = oristate_dict[bn_runing_mean_name]
                    state_dict[name_base + bn_running_var_name] = oristate_dict[bn_running_var_name]
                    state_dict[name_base + bn_num_batches_tracked_name] = \
                    oristate_dict[bn_num_batches_tracked_name]
                else:
                    # logger.info('yes yes')
                    state_dict[name_base + conv_weight_name] = oriweight
                    last_select_index = None

                    # 加载bn层
                    state_dict[name_base + bn_weight_name] = oristate_dict[bn_weight_name]
                    state_dict[name_base + bn_bias_name] = oristate_dict[bn_bias_name]
                    state_dict[name_base + bn_runing_mean_name] = oristate_dict[bn_runing_mean_name]
                    state_dict[name_base + bn_running_var_name] = oristate_dict[bn_running_var_name]
                    state_dict[name_base + bn_num_batches_tracked_name] = \
                        oristate_dict[bn_num_batches_tracked_name]

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base + conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']
        elif isinstance(module, nn.BatchNorm2d):
            if name not in all_bn_weight:
                logger.info('name: {}'.format(name))
                state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name_base + name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name_base + name + '.running_var'] = oristate_dict[name + '.running_var']
                state_dict[name_base + name + '.num_batches_tracked'] = oristate_dict[name + '.num_batches_tracked']

    model.load_state_dict(state_dict)

def load_resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        20: [3, 3, 3],
        32: [5, 5, 5],
        56: [9, 9, 9],
        80: [13, 13, 13],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    # logger.info('loading reserve ci from: ' + prefix + str(cov_id) + subfix)
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    # select_index = np.argsort(-ci)[orifilter_num - currentfilter_num:]  # preserved filter id

                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    # logger.info('yes yes')
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']
        # elif isinstance(module, nn.BatchNorm2d):
        #     logger.info('name: {}'.format(name))
        #     state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']
        #     state_dict[name_base + name + '.running_mean'] = oristate_dict[name + '.running_mean']
        #     state_dict[name_base + name + '.running_var'] = oristate_dict[name + '.running_var']
        #     state_dict[name_base + name + '.num_batches_tracked'] = oristate_dict[name + '.num_batches_tracked']

    model.load_state_dict(state_dict)

def load_resnet_tinyimagenet_56_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        20: [3, 3, 3],
        56: [9, 9, 9],
        80: [13, 13, 13],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                shape1 = state_dict[name_base+conv_name].size()
                shape2 = oristate_dict[conv_name].size()
                logger.info('shape1 is {}, shape2 is {}'.format(shape1, shape2))
                if shape1 == shape2:
                    state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_resnet562resnettinyimagenet56_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                new_model_shape = state_dict[name_base+conv_name].shape
                old_model_shape = oristate_dict[conv_name].shape
                if new_model_shape == old_model_shape:
                    state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter3resnet2adapter3resnet_tinyimagenet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir + '/ci_conv'
    subfix = ".npy"

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base + conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base + conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                new_model_shape = state_dict[name_base + conv_name].shape
                old_model_shape = oristate_dict[conv_name].shape
                if new_model_shape == old_model_shape:
                    state_dict[name_base + conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_adapter1resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.convdown'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.convup'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_adapter3resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_adapter3resnet_tinyimagenet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter3resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter5resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 1:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 3:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter6resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv1'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.conv2'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter8resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [13, 13, 13],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter9resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter10resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter11resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter12resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter13resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter14resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter15resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        20: [3, 3, 3],
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        # print(name)
        # print(module)
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter17resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter18resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter19resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter20resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter21resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter22resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter23resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            # convnum = 0
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def graf_load_adapter1resnet_model(args, model, oristate_dict, layer, logger, name_base=''):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            convnum = 0
            for l in range(4):
                cnt += 1
                cov_id = cnt
                conv_weight_name = None
                if l == 0 or l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(convnum + 1)
                    convnum += 1
                    conv_weight_name = conv_name + '.weight'
                elif l == 1:
                    conv_name = layer_name + str(k) + '.adapter' + '.convdown'
                    conv_weight_name = conv_name + '.weight'
                elif l == 2:
                    conv_name = layer_name + str(k) + '.adapter' + '.convup'
                    conv_weight_name = conv_name + '.weight'

                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                logger.info(conv_weight_name)
                if orifilter_num != currentfilter_num:
                    logger.info('origilter num: {}, currentfilter num: {}'.format(orifilter_num, currentfilter_num))
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    logger.info('ci shape: {}'.format(ci.shape))
                    logger.info('delta: {}'.format(orifilter_num - currentfilter_num))
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    logger.info(select_index)

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]
        # 嫁接载入权重全连接层不需要改变
        # elif isinstance(module, nn.Linear):
        #     state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
        #     state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def overall_load_arch_model(args, model, origin_model, ckpt, logger, graf=False):
    # 首先得到原始模型参数
    origin_model_arch = args.arch if graf == False else args.pretrained_arch
    print('overall origin model arch: ', origin_model_arch)
    if origin_model_arch == 'resnet_110':
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt['state_dict'], strict=False)
    oristate_dict = origin_model.state_dict()
    if graf == True:
        if 'resnet_56' in args.pretrained_arch:
            logger.info('overall load resnet model')
            overall_load_resnet_model(args, model, oristate_dict, 56, logger)
        else:
            raise
# 将原始模型的参数载入到压缩模型之中
def load_arch_model(args, model, origin_model, ckpt, logger, graf=False):
    # 首先得到原始模型参数
    origin_model_arch = args.arch if graf == False else args.pretrained_arch
    logger.info('origin model arch: {}'.format(origin_model_arch))
    # logger.info('original model arch')
    # logger.info(origin_model)
    if origin_model_arch == 'resnet_110':
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt['state_dict'], strict=False)
    oristate_dict = origin_model.state_dict()
    # 当在同一个模型上进行裁剪时
    if graf == False:
        # 将原始模型参数载入到压缩模型中
        if args.arch == 'vgg_16_bn':
            logger.info('load_vgg_model')
            load_vgg_model(args, model, oristate_dict, logger)
        elif args.arch == 'adapter_vgg_16_bn':
            logger.info('load_adapter_vgg_model')
            load_vgg_model(args, model, oristate_dict, logger)
        elif args.arch == 'adapter_vgg_16_bn_v4':
            logger.info('load_adapter_vgg_v4_model')
            load_vgg_model(args, model, oristate_dict, logger)
        elif args.arch == 'resnet_20':
            logger.info('load_resnet_20_model')
            load_resnet_model(args, model, oristate_dict, 20, logger)
        elif args.arch == 'resnet_32':
            logger.info('load_resnet_32_model')
            load_resnet_model(args, model, oristate_dict, 32, logger)
        elif args.arch == 'resnet_56':
            logger.info('load_resnet_model')
            load_resnet_model(args, model, oristate_dict, 56, logger)
        elif args.arch == 'resnet_80':
            logger.info('load_resnet_model')
            load_resnet_model(args, model, oristate_dict, 80, logger)
        elif args.arch == 'resnet_tinyimagenet_56':
            logger.info('load_resnet_tinyimagenet_56_model')
            load_resnet_tinyimagenet_56_model(args, model, oristate_dict, 56, logger)
        elif args.arch == 'resnet_110':
            logger.info('load_resnet_model')
            load_resnet_model(args, model, oristate_dict, 110, logger)
        elif args.arch == 'adapter1resnet_56':
            logger.info('load_adapter1resnet_model')
            load_adapter1resnet_model(args, model, oristate_dict, 56, logger)
        elif args.arch == 'adapter3resnet_56':
            logger.info('load_adapter3resnet_model')
            load_adapter3resnet_model(args, model, oristate_dict, 56, logger)
        elif args.arch == 'adapter3resnet_tinyimagenet_56':
            logger.info('adapter3resnet_tinyimagenet_56')
            load_adapter3resnet_tinyimagenet_model(args, model, oristate_dict, 56, logger)
        elif args.arch == 'adapter15resnet_20' or args.arch == 'adapter19resnet_20':
            logger.info('load adapter15 resnet 20 model')
            load_resnet_model(args, model, oristate_dict, 20, logger)
        elif args.arch == 'adapter16resnet_32':
            logger.info('load adapter16 resnet 32 model')
            load_resnet_model(args, model, oristate_dict, 32, logger)
        elif args.arch == 'adapter15resnet_32':
            logger.info('load adapter15 resnet 32 model')
            load_resnet_model(args, model, oristate_dict, 32, logger)
        elif 'resnet_56' in args.arch:
            logger.info('load resnet 56 model')
            load_resnet_model(args, model, oristate_dict, 56, logger)
        else:
            raise
    # 在不同的模型或者不同的数据集上进行裁剪
    elif graf == True:
        if args.pretrained_arch == args.finetune_arch:
            if args.finetune_arch == 'resnet_20':
                logger.info('graf_load_resnet20_model')
                graf_load_resnet_model(args, model, oristate_dict, 20, logger)
            elif args.finetune_arch == 'resnet_56':
                logger.info('graf_load_resnet56_model')
                graf_load_resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'resnet_80':
                logger.info('graf_load_resnet_model')
                graf_load_resnet_model(args, model, oristate_dict, 80, logger)
            elif args.finetune_arch == 'vgg_16_bn':
                logger.info('graf_load_vgg_16_bn_model')
                graf_load_vgg_model(args, model, oristate_dict, logger)
                # graf_load_resnet_model(args, model, oristate_dict, 80, logger)
            elif args.finetune_arch == 'adapter_vgg_16_bn':
                logger.info('graf_load_adapter_vgg_16_bn_model')
                graf_load_vgg_model(args, model, oristate_dict, logger)
                # graf_load_resnet_model(args, model, oristate_dict, 80, logger)
            elif args.finetune_arch == 'adapter15resnet_20':
                logger.info('graf_load_adapter_vgg_16_bn_model')
                graf_load_adapter15resnet_model(args, model, oristate_dict, 20, logger)
            elif args.finetune_arch == 'adapter1resnet_56':
                logger.info('graf_load_adapter1resnet_model')
                graf_load_adapter1resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter3resnet_56':
                logger.info('graf_load_adapter3resnet_model')
                graf_load_adapter3resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter5resnet_56':
                logger.info('graf_load_adapter5resnet_model')
                graf_load_adapter5resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter6resnet_56':
                logger.info('graf_load_adapter6resnet_model')
                graf_load_adapter6resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter8resnet_56':
                logger.info('graf_load_adapter8resnet_model')
                graf_load_adapter8resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter9resnet_56':
                logger.info('graf_load_adapter9resnet_model')
                graf_load_adapter9resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter10resnet_56':
                logger.info('graf_load_adapter10resnet_model')
                graf_load_adapter10resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter11resnet_56':
                logger.info('graf_load_adapter11resnet_model')
                graf_load_adapter11resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter12resnet_56':
                logger.info('graf_load_adapter10resnet_model')
                graf_load_adapter12resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter13resnet_56':
                logger.info('graf_load_adapter13resnet_model')
                graf_load_adapter13resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter14resnet_56':
                logger.info('graf_load_adapter14resnet_model')
                graf_load_adapter14resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter15resnet_56':
                logger.info('graf_load_adapter15resnet_model')
                graf_load_adapter15resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter17resnet_56':
                logger.info('graf_load_adapter17resnet_model')
                graf_load_adapter17resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter18resnet_56':
                logger.info('graf_load_adapter18resnet_model')
                graf_load_adapter18resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter19resnet_56':
                logger.info('graf_load_adapter19resnet_model')
                graf_load_adapter19resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter20resnet_56':
                logger.info('graf_load_adapter20resnet_model')
                graf_load_adapter20resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter21resnet_56':
                logger.info('graf_load_adapter21resnet_model')
                graf_load_adapter21resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter22resnet_56':
                logger.info('graf_load_adapter22resnet_model')
                graf_load_adapter22resnet_model(args, model, oristate_dict, 56, logger)
            elif args.finetune_arch == 'adapter23resnet_56':
                logger.info('graf_load_adapter23resnet_model')
                graf_load_adapter23resnet_model(args, model, oristate_dict, 56, logger)
            else:
                raise
        else:
            if args.pretrained_arch == 'resnet_56' and args.finetune_arch == 'resnet_tinyimagenet_56':
                logger.info('graf_load_resnet562resnettinyimagenet56_model')
                graf_load_resnet562resnettinyimagenet56_model(args, model, oristate_dict, 56, logger)
            elif args.pretrained_arch == 'adapter3resnet_56' and args.finetune_arch == 'adapter3resnet_tinyimagenet_56':
                logger.info('graf_load_adapter3resnet2adapter3resnettinyimagenet_model')
                graf_load_adapter3resnet2adapter3resnet_tinyimagenet_model(args, model, oristate_dict, 56, logger)
            elif args.pretrained_arch == 'supcon_adapter15resnet_56' and args.finetune_arch == 'adapter15resnet_56':
                logger.info('supcon_adapter15resnet to adapter15resnet')
                graf_load_adapter15resnet_model(args, model, oristate_dict, 56, logger)
            elif args.pretrained_arch == 'sl_mlp_resnet_56' and args.finetune_arch == 'resnet_56':
                logger.info('sl_mlp_resnet to resnet')
                graf_load_resnet_model(args, model, oristate_dict, 56, logger)
            elif args.pretrained_arch == 'sl_mlp_adapter15resnet_56' and args.finetune_arch == 'adapter15resnet_56':
                logger.info('sl_mlp_adapter15resnet to resnet')
                graf_load_adapter15resnet_model(args, model, oristate_dict, 56, logger)
            else:
                raise
# 学习率调整
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
        if epoch in [100, 150]:
            lr = cur_lr / 10

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        # lr = args.learning_rate
        # return
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
    if epoch < 2:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))

def train(epoch, train_loader, model, criterion, optimizer, args, logger, print_freq, device, scheduler = None):
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

        adjust_learning_rate(optimizer, epoch, i, num_iter, args, logger)

        # logits = model(images)
        # loss = criterion(logits, target)

        # compute outputy
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(images.size()[0]).to(device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # compute output
            logits = model(images)
            loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
        elif args.mixup_alpha > 0:
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            batch_size = images.size()[0]
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index, :]
            y_a, y_b = target, target[index]
            logits = model(mixed_images)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            # logits, _ = model(images)
            logits = model(images)
            loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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
            # print(i)
            images = images.to(device)
            target = target.to(device)

            # compute output
            # logits, _ = model(images)
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

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

# 保存相关变量到tensorboard中
def logstore(writer, train_losses, train_accuracy, test_losses, test_accuracy, epoch):
    writer.add_scalar('losses/train losses', train_losses, epoch)
    writer.add_scalar('accuracy/train accuracy', train_accuracy, epoch)
    writer.add_scalar('losses/test losses', test_losses, epoch)
    writer.add_scalar('accuracy/test accuracy', test_accuracy, epoch)

# 保存相关变量到tensorboard中
def lossstore(writer, train_losses, selfsupcon_losses, supcon_losses, epoch):
    writer.add_scalar('losses/train losses', train_losses, epoch)
    writer.add_scalar('losses/selfsupcon losses', selfsupcon_losses, epoch)
    writer.add_scalar('losses/supcon losses', supcon_losses, epoch)
