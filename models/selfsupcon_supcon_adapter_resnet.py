import torch.nn as nn
import torch.nn.functional as F
from .adapter import Adapter, Adapter3, Adapter4, Adapter5

"""
= =  =  =  =  =  =  +  +  +  +  +  =  =
= =  =  =  =  =  =  =  =  =  =  =  =  =
= =  +  +  +  =  =  =  =  =  =  =  =  =
= +  =  =  =  =  =  =  =  =  =  =  =  =
= =  =  =  =  +  =  =  +  +  =  =  =  =
= =  +  +  +  =  =  =  =  =  =  =  =  =
= =  =  =  =  =  =  =  =  =  =  =  =  =
= +  =  =  =  =  =  =  =  =  =  =  =  =
+ =  +  +  +  +  +  +  +  =  +  +  +  +
9 10 11 12 13 14 15 17 18 19 20 21 22 23
"""

adoch_cfg = {
    'adapter9': [16 * 9] + [32 * 9] + [64 * 9],
    'adapter10': [16 * 9] * 2 + [32 * 9] * 2 + [64 * 9] * 2,
    'adapter11': [16 * 9] * 3 + [32 * 9] * 3 + [64 * 9] * 3,
    'adapter12': [16 * 8] * 3 + [32 * 8] * 3 + [64 * 8] * 3,
    'adapter13': [64 * 8] * 3,
    'adapter14': [64 * 8] * 2,
    # 'adapter15': [64 * 8],
    # 'adapter15': [64 / 4],
    'adapter15': [64 / 1],
    'adapter16': [64 * 16],
    'adapter17': [64 * 8] * 2,
    'adapter18': [64 * 8] * 3,
    'adapter19': [64 * 8] * 2,
    'adapter20': [32 * 8] * 2 + [64 * 8] * 2,
    'adapter21': [16 * 8] * 2 + [32 * 8] * 2 + [64 * 8] * 2,
    'adapter22': [16 * 8] + [32 * 8] + [64 * 8],
    'adapter23': [32 * 8] + [64 * 8],
    'adapter24': [32 * 8]
}

nd_cfg = {
    # 在第几层加，从0开始
    'adapter9': [8],
    'adapter10': [3, 7],
    'adapter11': [2, 5, 8],
    'adapter12': [2, 5, 8],
    'adapter13': [2, 5, 8],
    'adapter14': [4, 8],
    'adapter15': [8],
    'adapter16': [8],
    'adapter17': [0, 8],
    'adapter18': [0, 4, 8],
    'adapter19': [0, 4],
    'adapter20': [0, 8],
    'adapter21': [0, 8],
    'adapter22': [8],
    'adapter23': [8],
    'adapter24': [8]
}

nd_stage = {
    # 在第几个stage加, 从1开始
    'adapter13': [3],
    'adapter14': [3],
    'adapter15': [3],
    'adapter16': [3],
    'adapter17': [3],
    'adapter18': [3],
    'adapter19': [3],
    'adapter20': [2, 3],
    'adapter21': [1, 2, 3],
    'adapter22': [1, 2, 3],
    'adapter23': [2, 3],
    'adapter24': [2]
}

adoch_20_cfg = {
    'adapter15': [64 * 8],
    'adapter16': [32 * 8] + [64 * 8],
    'adapter17': [16 * 8] + [32 * 8] + [64 * 8],
    'adapter18': [16 * 8] + [32 * 8],
    'adapter19': [32 * 8]
}

nd_20_cfg = {
    'adapter15': [2],
    'adapter16': [2],
    'adapter17': [2],
    'adapter18': [2],
    'adapter19': [2]
}

nd_20_stage = {
    'adapter15': [3],
    'adapter16': [2, 3],
    'adapter17': [1, 2, 3],
    'adapter18': [1, 2],
    'adapter19': [2]
}

def adapt_channel(sparsity, num_layers, adapter_sparsity, adapter_out_channel):
    print('adapter_sparsity: ', adapter_sparsity)
    if num_layers == 20:
        stage_repeat = [3, 3, 3]
        stage_out_channel = [16] + [16] * 3 + [32] * 3 + [64] * 3
    elif num_layers == 56:
        stage_repeat = [9, 9, 9]
        # 每一个stage的输出通道数，第一个stage为16，第二个为32最后一个stage为64
        stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9
        # adapter3resnet和adapter1resnet的时候用这个
        # adapter_out_channel = [4] * 9 + [8] * 9 + [16] * 9
        # adapter4resnet的时候用这个
        # adapter_out_channel = [64] * 9 + [128] * 9 + [256] * 9
        # adapter_out_channel = [32] * 9 + [64] * 9 + [128] * 9
        # adapter8resnet的时候用这个
        # adapter_out_channel = [16 * 9] * 4 + [32 * 9] * 4 + [64 * 9] * 4

    elif num_layers == 110:
        stage_repeat = [18, 18, 18]
        stage_out_channel = [16] + [16] * 18 + [32] * 18 + [64] * 18

    # 每一层的裁剪率
    stage_oup_cprate = []
    stage_oup_cprate += [sparsity[0]]
    for i in range(len(stage_repeat) - 1):
        stage_oup_cprate += [sparsity[i + 1]] * stage_repeat[i]
    stage_oup_cprate += [0.] * stage_repeat[-1]
    mid_cprate = sparsity[len(stage_repeat):]

    # overall_channel为每一个block的输出通道数
    overall_channel = []
    # mid_channel为每一个block的中间层输出通道数
    mid_channel = []
    # adapter channel
    adapter_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1 - mid_cprate[i - 1]))]
    for i in range(len(adapter_out_channel)):
        adapter_channel += [int((1 - adapter_sparsity[i]) * adapter_out_channel[i])]

    print('overall_channel: ', overall_channel)
    print('mid_channel: ', mid_channel)
    print('adapter_channel: ', adapter_channel)

    # return overall_channel, mid_channel, adapter_channel
    return overall_channel, mid_channel, adapter_channel

def resnet_adapt_channel(sparsity, num_layers):
    repeat = (num_layers - 2) // 6
    stage_repeat = [repeat] * 3
    stage_out_channel = [16] + [16] * repeat + [32] * repeat + [64] * repeat

    # 每一层的裁剪率
    stage_oup_cprate = []
    # 第一层的裁剪率直接就是sparsity的第一项
    stage_oup_cprate += [sparsity[0]]
    # 计算接下来两个stage的裁剪率，最后一个stage输出通道数不能进行裁剪
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [sparsity[i+1]] * stage_repeat[i]
    stage_oup_cprate += [0.] * stage_repeat[-1]
    mid_cprate = sparsity[len(stage_repeat):]
    # overall_channel为每一个block的输出通道数
    overall_channel = []
    # mid_channel为每一个block的中间层输出通道数
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]

    print('overall_channel: ', overall_channel)
    print('mid_channel: ', mid_channel)
    return overall_channel, mid_channel

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


expansion = 1


class BasicBlock(nn.Module):
    expansion = 1

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

        self.shortcut = nn.Sequential()
        # x shape is (N, C, H, W)
        if stride != 1 or inplanes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    # ::2表示隔一个数字取一个，这样形状大小就减半
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (
                                    0, 0, 0, 0, (planes - inplanes) // 2, planes - inplanes - (planes - inplanes) // 2),
                                    "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (
                                    0, 0, 0, 0, (planes - inplanes) // 2, planes - inplanes - (planes - inplanes) // 2),
                                    "constant", 0))
            # self.shortcut = LambdaLayer(
            #    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),"constant", 0))

            '''self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                #nn.BatchNorm2d(planes),
            )#'''

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10, dataset=None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.dataset = dataset
        self.overall_channel, self.mid_channel = resnet_adapt_channel(sparsity, num_layers)

        self.layer_num = 0
        # 训练dtd的baseline的时候用这个
        if self.dataset == 'dtd':
            # self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=5, stride=1, padding=2,
            #                        bias=False)
            self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        self.layer_num += 1

        #self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        indim = 128

        self.selfsupconhead = nn.Sequential(
            nn.Linear(64, indim),
            nn.ReLU(inplace=True),
            nn.Linear(indim, 64)
        )

        self.supconhead = nn.Sequential(
            nn.Linear(64, indim),
            nn.ReLU(inplace=True),
            nn.Linear(indim, 64)
        )

        # if self.num_layer == 56:
        #     self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        # else:
        #     self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)


    def _make_layer(self, block, blocks_num, stride):
        layers = []
        # block中的参数分别为中间层输出通道，block输入通道数，block输出通道数
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                 self.overall_channel[self.layer_num], stride))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num]))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.dataset == 'dtd':
            x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        selfsupcon_x = self.selfsupconhead(x)
        selfsupcon_x = F.normalize(selfsupcon_x, dim=1)

        supcon_x = self.supconhead(x)
        supcon_x = F.normalize(supcon_x, dim=1)

        return selfsupcon_x, supcon_x, x
        # return selfsupcon_x, supcon_x

class ResNet_New(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10, adapter_sparsity=None,
                 adapter_out_channel=None, need_adapter=None, need_stage=None):
        super(ResNet_New, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.need_adapter = need_adapter
        self.need_stage = [1, 2, 3]
        if need_stage is not None:
            self.need_stage = need_stage
        self.overall_channel, self.mid_channel, self.adapter_channel = adapt_channel(
            sparsity, num_layers, adapter_sparsity, adapter_out_channel)

        self.layer_num = 0
        self.adapter_layer = 0
        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        # self.layers = nn.ModuleList()
        self.layer_num += 1

        # self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1, stage=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2, stage=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2, stage=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        indim = 128

        self.selfsupconhead = nn.Sequential(
            nn.Linear(64, indim),
            nn.ReLU(inplace=True),
            nn.Linear(indim, 64)
        )

        self.supconhead = nn.Sequential(
            nn.Linear(64, indim),
            nn.ReLU(inplace=True),
            nn.Linear(indim, 64)
        )

        # if self.num_layer == 56:
        #     self.fc = nn.Linear(64 * expansion, num_classes)
        # else:
        #     self.linear = nn.Linear(64 * expansion, num_classes)

    def _make_layer(self, block, blocks_num, stride, stage):
        layers = []
        # block中的参数分别为中间层输出通道，block输入通道数，block输出通道数
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                            self.overall_channel[self.layer_num], stride))
        self.layer_num += 1
        for i in range(1, blocks_num):
            # 当该层是basicblock层时
            if i in self.need_adapter and stage in self.need_stage:
                layers.append(Adapter3(self.overall_channel[self.layer_num - 1], self.adapter_channel[self.adapter_layer]))
                self.adapter_layer += 1
            else:
                layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                    self.overall_channel[self.layer_num]))

            # if i not in self.need_adapter:
            #     layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
            #                         self.overall_channel[self.layer_num]))
            # 当该层是adapter层时
            # else:
            #     layers.append(Adapter3(self.overall_channel[self.layer_num - 1], self.adapter_channel[self.adapter_layer]))
            #     self.adapter_layer += 1
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        selfsupcon_x = self.selfsupconhead(x)
        selfsupcon_x = F.normalize(selfsupcon_x, dim=1)

        supcon_x = self.supconhead(x)
        supcon_x = F.normalize(supcon_x, dim=1)

        # if self.num_layer == 56:
        #     x = self.fc(x)
        # else:
        #     x = self.linear(x)

        # return x, feat
        # return x, x
        # return selfsupcon_x, supcon_x, x
        return selfsupcon_x, supcon_x


class ResNet_New_New(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10, adapter_sparsity=None,
                 adapter_out_channel=None, need_adapter=None, need_stage=None, dataset=None):
        super(ResNet_New_New, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.need_adapter = need_adapter
        self.need_stage = [1, 2, 3]
        if need_stage is not None:
            self.need_stage = need_stage
        self.overall_channel, self.mid_channel, self.adapter_channel = adapt_channel(
            sparsity, num_layers, adapter_sparsity, adapter_out_channel)

        self.layer_num = 0
        self.adapter_layer = 0
        self.dataset = dataset
        if self.dataset == 'dtd':
            # self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=5, stride=1, padding=2,
            #                        bias=False)
            self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                                   bias=False)
        # self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=5, stride=1, padding=2,
        #                        bias=False)
        # self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        # self.layers = nn.ModuleList()
        self.layer_num += 1

        # self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1, stage=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2, stage=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2, stage=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # if self.num_layer == 56:
        #     self.fc = nn.Linear(64 * expansion, num_classes)
        # else:
        #     self.linear = nn.Linear(64 * expansion, num_classes)
        # self.head = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 64)
        # )

    def _make_layer(self, block, blocks_num, stride, stage):
        layers = []
        if stage in self.need_stage and 0 in self.need_adapter:
            layers.append(Adapter5(self.overall_channel[self.layer_num - 1], self.adapter_channel[self.adapter_layer]
                                   , self.overall_channel[self.layer_num], stride=stride))
            self.adapter_layer += 1
        else:
        # block中的参数分别为中间层输出通道，block输入通道数，block输出通道数
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                self.overall_channel[self.layer_num], stride))
        self.layer_num += 1
        for i in range(1, blocks_num):
            # 当该层是basicblock层时
            if i in self.need_adapter and stage in self.need_stage:
                layers.append(Adapter5(self.overall_channel[self.layer_num - 1], self.adapter_channel[self.adapter_layer]
                                       , self.overall_channel[self.layer_num - 1]))
                self.adapter_layer += 1
            else:
                layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                    self.overall_channel[self.layer_num]))

            # if i not in self.need_adapter:
            #     layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
            #                         self.overall_channel[self.layer_num]))
            # 当该层是adapter层时
            # else:
            #     layers.append(Adapter3(self.overall_channel[self.layer_num - 1], self.adapter_channel[self.adapter_layer]))
            #     self.adapter_layer += 1
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.dataset == 'dtd':
            x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feat = F.normalize(self.head(x), dim=1)
        # feat = F.normalize(x, dim=1)
        # return feat
        # feature = x.clone()

        # if self.num_layer == 56:
        #     x = self.fc(x)
        # else:
        #     x = self.linear(x)

        return x, feat
        # return x

# def adapter15resnet_56(sparsity, num_classes, adapter_sparsity, dataset=None):
#     return ResNet_New(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity,
#                       adapter_out_channel=adoch_cfg['adapter15'], need_adapter=nd_cfg['adapter15'],
#                       need_stage=nd_stage['adapter15'])

def supcon_adapter15resnet_56(sparsity, num_classes, adapter_sparsity, dataset=None):
    return ResNet_New(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity,
                      adapter_out_channel=adoch_cfg['adapter15'], need_adapter=nd_cfg['adapter15'],
                      need_stage=nd_stage['adapter15'])

def selfsupcon_adapter15resnet_56(sparsity, num_classes, adapter_sparsity, dataset=None):
    return ResNet_New(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity,
                      adapter_out_channel=adoch_cfg['adapter15'], need_adapter=nd_cfg['adapter15'],
                      need_stage=nd_stage['adapter15'])

def selfsupcon_supcon_adapter15resnet_56(sparsity, num_classes, adapter_sparsity, dataset=None):
    return ResNet_New(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity,
                      adapter_out_channel=adoch_cfg['adapter15'], need_adapter=nd_cfg['adapter15'],
                      need_stage=nd_stage['adapter15'])

def selfsupcon_supcon_adapter15resnet_20(sparsity, num_classes, adapter_sparsity, dataset=None):
    return ResNet_New(BasicBlock, 20, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity,
                      adapter_out_channel=adoch_20_cfg['adapter15'], need_adapter=nd_20_cfg['adapter15'],
                      need_stage=nd_20_stage['adapter15'])

def selfsupcon_supcon_resnet_56(sparsity, num_classes, dataset=None):
    return ResNet(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes)


