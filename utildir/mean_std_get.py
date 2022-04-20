'''Get dataset mean and std with PyTorch.'''
from __future__ import print_function

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import models
import utils
import time

# from models import *
from data import cifar10, cifar100, tinyimagenet, svhn, dtd
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='dtd', type=str, help='dataset')
parser.add_argument('--batch_size', default='64', type=int, help='dataset')
parser.add_argument('--data_dir', default='/Users/chenjie/dataset/dtd', type=str, help='dataset')
parser.add_argument('--split', default='1', type=str, help='dataset')

args = parser.parse_args()

# train_loader, valid_loader = cifar100.load_cifar_data(args)
# train_loader, valid_loader = tinyimagenet.load_tinyimagenet_data(args)
# train_loader, valid_loader = svhn.load_svhn_data(args)
train_loader, valid_loader = dtd.load_dtd_data(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
h, w = 0, 0
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum/len(train_loader.dataset)/h/w
print('mean: %s' % mean.view(-1))

chsum = None
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
std = torch.sqrt(chsum/(len(train_loader.dataset) * h * w - 1))
print('std: %s' % std.view(-1))

print('Done!')