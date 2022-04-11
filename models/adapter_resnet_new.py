
import torch.nn as nn
import torch.nn.functional as F
from .adapter import Adapter, Adapter3, Adapter4

def adapt_channel(sparsity, num_layers, adapter_sparsity):
    print('adapter_sparsity: ', adapter_sparsity)
    if num_layers==56:
        stage_repeat = [9, 9, 9]
        # 每一个stage的输出通道数，第一个stage为16，第二个为32最后一个stage为64
        stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9
        # adapter3resnet和adapter1resnet的时候用这个
        # adapter_out_channel = [4] * 9 + [8] * 9 + [16] * 9
        # adapter4resnet的时候用这个
        adapter_out_channel = [64] * 9 + [128] * 9 + [256] * 9
        # adapter_out_channel = [32] * 9 + [64] * 9 + [128] * 9

    elif num_layers==110:
        stage_repeat = [18, 18, 18]
        stage_out_channel = [16] + [16] * 18 + [32] * 18 + [64] * 18

    # 每一层的裁剪率
    stage_oup_cprate = []
    stage_oup_cprate += [sparsity[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [sparsity[i+1]] * stage_repeat[i]
    stage_oup_cprate += [0.] * stage_repeat[-1]
    mid_cprate = sparsity[len(stage_repeat):]

    # overall_channel为每一个block的输出通道数
    overall_channel = []
    # mid_channel为每一个block的中间层输出通道数
    mid_channel = []
    # adapter channel
    adapter_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]
    for i in range(len(adapter_out_channel)):
        adapter_channel += [int((1 - adapter_sparsity[i]) * adapter_out_channel[i])]

    print('overall_channel: ', overall_channel)
    print('mid_channel: ', mid_channel)
    print('adapter_channel: ', adapter_channel)

    return overall_channel, mid_channel, adapter_channel


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

class BasicAdptBlock1(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1, adapterplanes=None):
        super(BasicAdptBlock1, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        # if adapterplanes is None:
        #     adapterplanes = midplanes
        # adapter结构
        self.adapter = Adapter(midplanes, adapterplanes)

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

        # 运用adapter结构
        out = self.adapter(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class BasicAdptBlock2(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1):
        super(BasicAdptBlock2, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        # adapter层
        self.adapter1 = Adapter(midplanes, midplanes // 4)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, planes)
        # adapter层
        self.adapter2 = Adapter(planes, planes // 4)
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
        # 通过adapter层输出
        out = self.adapter1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.adapter2(out)
        out = self.bn2(out)

        #print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class BasicAdptBlock3(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1, adapterplanes=None):
        super(BasicAdptBlock3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.adapter = Adapter3(midplanes, adapterplanes)

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

        # 通过adapter层输出
        out = self.adapter(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class BasicAdptBlock5(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1, adapterplanes=None):
        super(BasicAdptBlock5, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        # midplanes和planes的大小一致
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.adapter = Adapter3(planes, adapterplanes)

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

        # print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        # 通过adapter层输出
        out = self.adapter(out)
        return out

class BasicAdptBlock6(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1, adapterplanes=None):
        super(BasicAdptBlock6, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.adapter = Adapter4(midplanes, adapterplanes)

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

        # 通过adapter层输出
        out = self.adapter(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

class BasicAdptBlock7(nn.Module):

    # 中间层输出通道，输入通道，输出通道
    def __init__(self, midplanes, inplanes, planes, stride=1, adapterplanes=None):
        super(BasicAdptBlock7, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.adapter = Adapter4(midplanes, adapterplanes)
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
        # print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        # 通过adapter层输出
        out = self.adapter(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10, adapter_sparsity=None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.overall_channel, self.mid_channel, self.adapter_channel = adapt_channel(sparsity, num_layers, adapter_sparsity)

        self.layer_num = 0
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
            self.fc = nn.Linear(64 * expansion, num_classes)
        else:
            self.linear = nn.Linear(64 * expansion, num_classes)


    def _make_layer(self, block, blocks_num, stride):
        layers = []
        # block中的参数分别为中间层输出通道，block输入通道数，block输出通道数
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                 self.overall_channel[self.layer_num], stride, self.adapter_channel[self.layer_num - 1]))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num], adapterplanes=self.adapter_channel[self.layer_num - 1]))
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

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x


def adapter1resnet_56(sparsity, num_classes, adapter_sparsity):
    return ResNet(BasicAdptBlock1, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity)

def adapter1resnet_110(sparsity, num_classes):
    return ResNet(BasicAdptBlock1, 110, sparsity=sparsity, num_classes=num_classes)

def adapter2resnet_56(sparsity, num_classes):
    return ResNet(BasicAdptBlock2, 56, sparsity=sparsity, num_classes=num_classes)

def adapter2resnet_110(sparsity, num_classes):
    return ResNet(BasicAdptBlock2, 110, sparsity=sparsity, num_classes=num_classes)

def adapter3resnet_56(sparsity, num_classes, adapter_sparsity):
    return ResNet(BasicAdptBlock3, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity)

def adapter5resnet_56(sparsity, num_classes, adapter_sparsity):
    return ResNet(BasicAdptBlock5, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity)

def adapter6resnet_56(sparsity, num_classes, adapter_sparsity):
    return ResNet(BasicAdptBlock6, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity)

def adapter7resnet_56(sparsity, num_classes, adapter_sparsity):
    return ResNet(BasicAdptBlock7, 56, sparsity=sparsity, num_classes=num_classes, adapter_sparsity=adapter_sparsity)

