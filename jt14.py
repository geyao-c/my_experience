import argparse
from thop import profile
from models.adapter_resnet_imagenet import adapter15resnet_34
import torch
import utils_append
from models.resnet_imagenet import resnet_34
from models.efficientnet import efficientnet_b0_changed_v4
from models.adapter_efficientnet import adapter_efficientnet_b0_changed_v5

parser = argparse.ArgumentParser("ImageNet training")
parser.add_argument('--data_dir', type=str, default='../data', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
args = parser.parse_args()

def fun1():
    sparsity = '[0.]+[0.2]*3+[0.4]*16'
    sparsity = utils_append.analysis_sparsity(sparsity)
    resnet_34_original = resnet_34([0.]*100)
    resnet_34_pruned = resnet_34(sparsity)
    input = torch.randn(1, 3, 224, 224)
    macs1, params1 = profile(resnet_34_original, inputs=(input,))
    macs2, params2 = profile(resnet_34_pruned, inputs=(input,))


    adapter_resnet_34_original = adapter15resnet_34([0.] * 100)
    adapter_resnet_34_pruned = adapter15resnet_34(sparsity, [0.4])
    input = torch.randn(1, 3, 224, 224)
    macs3, params3 = profile(adapter_resnet_34_original, inputs=(input,))
    macs4, params4 = profile(adapter_resnet_34_pruned, inputs=(input,))

    print('macs1: {}, params1: {}'.format(macs1, params1))
    print('macs2: {}, params2: {}'.format(macs2, params2))
    print('macs2/macs1: {}, params2/params1: {}'.format(1 - macs2 / macs1, 1 - params2 / params1))
    print('macs3: {}, params3: {}'.format(macs3, params3))
    print('macs4: {}, params4: {}'.format(macs4, params4))
    print('macs4/macs3: {}, params4/params3: {}'.format(1 - macs4 / macs3, 1 - params4 / params3))

def fun2():
    input = torch.randn(1, 3, 32, 32)

    sparsity1 = '[0.]+[0.25]+[0.3]*3+0.32+0.315+0.3'
    sparsity1 = utils_append.analysis_sparsity(sparsity1)
    model1 = efficientnet_b0_changed_v4([0.]*100)
    model2 = efficientnet_b0_changed_v4(sparsity1)
    macs1, params1 = profile(model1, inputs=(input,))
    macs2, params2 = profile(model2, inputs=(input,))

    sparsity2 = '[0.]+[0.25]+[0.3]*3+0.32+0.315+0.315+0.3'
    sparsity2 = utils_append.analysis_sparsity(sparsity2)
    model3 = adapter_efficientnet_b0_changed_v5([0.]*100, 10, [0.]*100)
    model4 = adapter_efficientnet_b0_changed_v5(sparsity2, 10, [0.]*100)
    macs3, params3 = profile(model3, inputs=(input,))
    macs4, params4 = profile(model4, inputs=(input,))

    print('macs1: {}, params1: {}'.format(macs1, params1))
    print('macs2: {}, params2: {}'.format(macs2, params2))
    print('macs2/macs1: {}, params2/params1: {}'.format(1 - macs2 / macs1, 1 - params2 / params1))
    print('macs3: {}, params3: {}'.format(macs3, params3))
    print('macs4: {}, params4: {}'.format(macs4, params4))
    print('macs4/macs3: {}, params4/params3: {}'.format(1 - macs4 / macs3, 1 - params4 / params3))

if __name__ == '__main__':
    fun1()
    # fun2()

# 278405196.0 3893206.0
# 280051836.0 3918982.0

# resnet_34 entir network
# 3651388416.0 21367348.0
# resnet_34 encoder
# 3651132416.0 21110848.0

# pruned resnet_34 entir network
# 1883516142.0 11776924.0
# pruned resnet_34 encoder
# 1883260142.0 11520424.0

