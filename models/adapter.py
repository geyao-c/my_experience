import torch.nn as nn
import torch.nn.functional as F

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


