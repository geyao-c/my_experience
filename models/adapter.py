import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# 先缩小然后变大
class Adapter(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels):
        super(Adapter, self).__init__()
        self.convdown = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.convup = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.convdown(x)
        out = self.relu(out)
        out = self.convup(out)
        return out + x

class Adapter3(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels):
        super(Adapter3, self).__init__()
        # 缩小
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 变大
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out

# 没有分支的adapter结构
class Adapter4(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels):
        super(Adapter4, self).__init__()
        # 缩小
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 变大
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = out + x
        out = self.relu2(out)
        return out

# 能够改变feature map H，W的adapter结构
class Adapter5(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels, out_channels, stride=1):
        super(Adapter5, self).__init__()
        # 缩小
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 变大
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        # pad函数的填充顺序为最后一层到第一层
        if stride != 1 or in_channels != out_channels:
            # 这里的stride非1即2
            if stride != 1:
                self.shortcut = LambdaLayer(
                    # ::2表示隔一个数字取一个，这样形状大小就减半
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (out_channels - in_channels) // 2,
                                                        out_channels - in_channels - (out_channels - in_channels) // 2),
                                    "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],(0, 0, 0, 0, (out_channels - in_channels) // 2,
                                                   out_channels - in_channels - (out_channels - in_channels) // 2),
                                    "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out

class VGG_Adapter(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGG_Adapter, self).__init__()
        # 缩小
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 变大
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio

        # if not squeeze_channels.is_integer():
        #     raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)

        # self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        # self.non_linear1 = Swish()
        # self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        # self.non_linear2 = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # y = torch.mean(x, (2, 3), keepdim=True)
        # y = self.non_linear1(self.se_reduce(y))
        # y = self.non_linear2(self.se_expand(y))
        # y = x * y

        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.conv[1](self.conv[0](y))
        y = self.conv[3](self.conv[2](y))
        y = x * y

        return y

class Efficientnet_Adapter(nn.Module):
    # 输入通道数，隐藏通道数
    def __init__(self, in_channels, hidden_channels):
        super(Efficientnet_Adapter, self).__init__()
        # 缩小
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 变大
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out

def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _DropPath(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
            # mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).to(device)
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)

    return x

class Adapter_MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(Adapter_MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        # 根据输入通道数和输出通道数是否相等决定是否拥有残差连接
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                _BatchNorm(expand_channels),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase, 点卷积
        # dw = nn.Sequential(
        #     nn.Conv2d(
        #         expand_channels,
        #         expand_channels,
        #         kernel_size,
        #         stride,
        #         kernel_size//2,
        #         groups=expand_channels,
        #         bias=False
        #     ),
        #     _BatchNorm(expand_channels),
        #     Swish()
        # )
        # conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            _BatchNorm(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + _DropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)


