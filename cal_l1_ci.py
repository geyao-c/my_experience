import numpy as np
import torch
import os
import argparse
import time
from models.resnet_imagenet import resnet_34
from models.adapter_resnet_imagenet import adapter15resnet_34

def getparse():
    parser = argparse.ArgumentParser("cal L1 norm")
    parser.add_argument('--arch', type=str, default='resnet_34', help='architecture')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--num_layers', type=int, default=None, help='conv layers in the model')
    parser.add_argument('--save_dir', type=str, default=None, help='ci saved dir')
    return parser.parse_args()

def cal_l1(convl):
    x = torch.sum(torch.abs(convl.view(convl.size(0), -1)), dim=1)
    return x

def cal_ci(model):
    layer_ci = []
    # 第一层卷积
    x = cal_l1(model.conv1.weight.data)
    layer_ci.append(x)

    for i in range(len(model.layer1)):
        x = cal_l1(model.layer1[i].conv1.weight.data); layer_ci.append(x)
        x = cal_l1(model.layer1[i].conv2.weight.data); layer_ci.append(x)

    for i in range(len(model.layer2)):
        x = cal_l1(model.layer2[i].conv1.weight.data); layer_ci.append(x)
        x = cal_l1(model.layer2[i].conv2.weight.data); layer_ci.append(x)

    for i in range(len(model.layer3)):
        x = cal_l1(model.layer3[i].conv1.weight.data); layer_ci.append(x)
        x = cal_l1(model.layer3[i].conv2.weight.data); layer_ci.append(x)

    for i in range(len(model.layer4)):
        x = cal_l1(model.layer4[i].conv1.weight.data); layer_ci.append(x)
        x = cal_l1(model.layer4[i].conv2.weight.data); layer_ci.append(x)

    return layer_ci

def main():
    # 首先获取参数
    args = getparse()

    # 加载模型
    model = eval(args.arch)()
    ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'], strict=False)

    # 计算L1-norm分数
    ci = cal_ci(model)

    for i in range(args.num_layers):
        print(i)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        np.save(args.save_dir + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
    main()

"""
conv1.weight
layer1.0.conv1.weight
"""