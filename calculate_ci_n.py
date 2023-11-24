import numpy as np
import torch
import os
import argparse
import time

"""
# adapter使用的层数
python calculate_ci_n.py --arch adapter3resnet_tinyimagenet_56 --repeat 5 --num_layers 109 \
--feature_map_dir ../conv_feature_map/56.2_adapter4resnet_tinyimagenet_56_tinyimagenet_repeat5
"""

parser = argparse.ArgumentParser(description='Calculate CI')
parser.add_argument('--arch', type=str, default='resnet_56', # choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
                    help='architecture to calculate feature maps')
parser.add_argument('--repeat', type=int, default=5, help='repeat times')
parser.add_argument('--num_layers', type=int, default=55, help='conv layers in the model')
parser.add_argument('--feature_map_dir', type=str, default='./conv_feature_map', help='feature maps dir')
parser.add_argument('--save_dir', type=str, default=None, help='ci saved dir')
parser.add_argument('--save_path', type=str, default=None, help='ci saved dir')

args = parser.parse_args()

def reduced_1_row_norm(input, row_index, data_index):
    # input shape is (H, C, H * W)
    # 把这一行赋值为0
    input[data_index, row_index, :] = torch.zeros(input.shape[-1])
    # 求矩阵的核范数
    m = torch.norm(input[data_index, :, :], p = 'nuc').item()
    return m

def ci_score(path_conv):
    # 保留4位小数
    conv_output = torch.tensor(np.round(np.load(path_conv), 4))
    # print(conv_output)
    print(conv_output.shape)
    # 参数的意义分别为: batch, channel, height, width
    # conv_reshape shape is (N, C, H * W)
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)

    r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
    for i in range(conv_reshape.shape[0]):
        for j in range(conv_reshape.shape[1]):
            r1_norm[i, j] = reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)

    ci = np.zeros_like(r1_norm)

    for i in range(r1_norm.shape[0]):
        original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        ci[i] = original_norm - r1_norm[i]

    # return shape: [batch_size, filter_number]
    return ci

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):
        repeat_ci_mean = []
        start = time.time()
        for i in range(repeat):
            print('num layer {}, repeat {}'.format(j, i))
            index = j * repeat + i + 1
            # add
            dirpath = args.feature_map_dir
            path_conv = os.path.join(dirpath, 'conv_feature_map_tensor({}).npy'.format(str(index)))
            # path_conv = "./conv_feature_map/{0}_repeat5/conv_feature_map_tensor({1}).npy".format(str(args.arch), str(index))
            # path_nuc = "./feature_conv_nuc/resnet_56_repeat5/feature_conv_nuctensor({0}).npy".format(str(index))
            # batch_ci = ci_score(path_conv, path_nuc)
            batch_ci = ci_score(path_conv)
            # 一个batch取平均值
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            repeat_ci_mean.append(single_repeat_ci_mean)
        # 5个batch取平均值
        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)
        end = time.time()
        print('layer shape is {}, time cost {:.3f}'.format(layer_ci_mean.shape, (end - start)))

    return np.array(layer_ci_mean_total)

def special_adapter15resnet_34(repeat):
    num_layers = [1, 2, 7, 8, 15, 16, 27, 28]
    layer_ci_mean_total = []
    for j in num_layers:
        repeat_ci_mean = []
        start = time.time()
        for i in range(repeat):
            print('num layer {}, repeat {}'.format(j, i))
            index = j * repeat + i + 1
            # add
            dirpath = args.feature_map_dir
            path_conv = os.path.join(dirpath, 'conv_feature_map_tensor({}).npy'.format(str(index)))
            batch_ci = ci_score(path_conv)
            # 一个batch取平均值
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            repeat_ci_mean.append(single_repeat_ci_mean)
        # 5个batch取平均值
        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)
        end = time.time()
        print('layer shape is {}, time cost {:.3f}'.format(layer_ci_mean.shape, (end - start)))

    return np.array(layer_ci_mean_total)

def special_resnet_50(repeat):
    num_layers = [1, 2, 3, 10, 11, 12, 22, 23, 24, 40, 41, 42]
    layer_ci_mean_total = []
    for j in num_layers:
        repeat_ci_mean = []
        start = time.time()
        for i in range(repeat):
            print('num layer {}, repeat {}'.format(j, i))
            index = j * repeat + i + 1
            # add
            dirpath = args.feature_map_dir
            path_conv = os.path.join(dirpath, 'conv_feature_map_tensor({}).npy'.format(str(index)))
            batch_ci = ci_score(path_conv)
            # 一个batch取平均值
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            repeat_ci_mean.append(single_repeat_ci_mean)
        # 5个batch取平均值
        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)
        end = time.time()
        print('layer shape is {}, time cost {:.3f}'.format(layer_ci_mean.shape, (end - start)))

    return np.array(layer_ci_mean_total)

def main():
    import sys

    # 打开一个文件以写入输出
    f = open("output.txt", "w")
    # 保存原来的标准输出
    original_stdout = sys.stdout
    # 重定向标准输出到文件
    sys.stdout = f

    repeat = args.repeat
    num_layers = args.num_layers

    # 构建save_path
    # 默认为从 "./calculated_ci开始"
    if args.save_dir is None:
        save_dir = './calculated_ci'
        # 根据feature_map_dir得到准确率和数据集
        # 提取出模型准确率
        fm_list = args.feature_map_dir.split('/')[2].split('_')
        model_accu = str(fm_list[0])
        dataset = None
        dst = ['cifar100', 'cifar10', 'tinyimagenet', 'svhn', 'mnist', 'cifar10andsvhn']
        for item in dst:
            if item in fm_list: dataset = item
        print('save_dir: ', save_dir, " model_accu: ", model_accu, " arch: ", args.arch, ' dataset: ' , dataset)
        save_dir = os.path.join(save_dir, model_accu + '_' + args.arch + '_' + dataset)
    else:
        save_dir = args.save_dir

    if args.arch == 'adapter15resnet_34':
        special_adapter15resnet_34(repeat)
    elif args.arch == 'resnet_50':
        special_resnet_50(repeat)
    else:
        # 计算ci
        ci = mean_repeat_ci(repeat, num_layers)

        if args.arch == 'resnet_50':
            num_layers = 53
        for i in range(num_layers):
            print(i)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_dir + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
    main()



