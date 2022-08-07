import torch.nn as nn
from collections import OrderedDict

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
#            [50, 50,      101, 101,      202, 202, 202,      128, 128, 128,      128, 128, 512]
#            [0.79, 0.79,  0.79, 0.79,    0.79,0.79,0.79      0.5, 0.5, 0.5,      0.5, 0.5, 1]
# [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
adapter_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512 * 8, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]

adapter_lid = [15, 16]
class VGG(nn.Module):
    def __init__(self, sparsity, cfg=None, num_classes=10):
        super(VGG, self).__init__()

        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg

        self.sparsity = sparsity[:]
        self.sparsity.append(0.0)

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-1], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt, acnt = 0, 0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            elif i in adapter_lid:
                x = int(x * (1-self.sparsity[cnt]))
                cnt += 1

                conv2d = nn.Conv2d(in_channels, x , kernel_size=1, padding=0)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x
            else:
                x = int(x * (1-self.sparsity[cnt]))

                cnt+=1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def adapter_vgg_16_bn(sparsity, adapter_sparsity = None, num_classes=10, dataset=None):
    return VGG(sparsity=sparsity, cfg=adapter_cfg, num_classes=num_classes)