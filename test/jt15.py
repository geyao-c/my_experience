import torch
from models.resnet_imagenet import resnet_50

model = resnet_50([0.] * 100)
torch.save(model, '../pretrained_models/1.0_resnet_50_imagenet.pth.tar')
