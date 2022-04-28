
import torch.nn as nn
import torch.nn.functional as F

def adapt_channel(sparsity, num_layers):
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
            if stride!=1:
                self.shortcut = LambdaLayer(
                    # ::2表示隔一个数字取一个，这样形状大小就减半
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            #self.shortcut = LambdaLayer(
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

        #print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
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
        self.overall_channel, self.mid_channel = adapt_channel(sparsity, num_layers)

        self.layer_num = 0
        # 训练dtd的baseline的时候用这个
        if self.dataset == 'dtd':
            self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=5, stride=1, padding=2,
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

        if self.num_layer == 56:
            self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        else:
            self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)


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
        # feature = x.clone()

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        # return x, feature
        return x

def resnet_56(sparsity, num_classes, dataset=None):
    # 不同的数据集网络结构可能有细微差别
    return ResNet(BasicBlock, 56, sparsity=sparsity, num_classes=num_classes, dataset=dataset)

def resnet_110(sparsity, num_classes):
    return ResNet(BasicBlock, 110, sparsity=sparsity, num_classes=num_classes)

def resnet_80(sparsity, num_classes):
    return ResNet(BasicBlock, 80, sparsity=sparsity, num_classes=num_classes)
