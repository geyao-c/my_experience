# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import utils_append
from models.resnet_imagenet import resnet_34
from models.adapter_resnet_imagenet import adapter15resnet_34

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    features, output = model(img)
    # features = model(img)
    # output = model.fc(features)
    print(features.shape)


    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    print(grads.shape)
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

if __name__ == '__main__':
    sparsity = '[0.]+[0.2]*3+[0.4]*16'
    adapter_sparsity = '[0.4]*1'
    sparsity = utils_append.analysis_sparsity(sparsity)
    adapter_sparsity = utils_append.analysis_sparsity(adapter_sparsity)

    resnet_34_model1 = resnet_34(sparsity)
    adapter15resnet_34_model1 = adapter15resnet_34(sparsity, adapter_sparsity)
    resnet_34_model2 = resnet_34(sparsity)
    adapter15resnet_34_model2 = adapter15resnet_34(sparsity, adapter_sparsity)

    resnet_34_ckpt1 = torch.load('../pretrained_models/77.124_pruned_resnet_34_imagenetA.pth.tar', map_location='cpu')
    adapter15resnet_34_ckpt1 = torch.load('../pretrained_models/77.972_pruned_adapter_resnet34_imagenetA.pth.tar', map_location='cpu')
    resnet_34_ckpt2 = torch.load('../pretrained_models/78.244_pruned_resnet34_imagenetB.pth.tar', map_location='cpu')
    adapter15resnet_34_ckpt2 = torch.load('../pretrained_models/79.028_pruned_adapter_renset34_imagenetB.pth.tar', map_location='cpu')

    resnet_34_model1.load_state_dict(resnet_34_ckpt1['state_dict'], strict=False)
    adapter15resnet_34_model1.load_state_dict(adapter15resnet_34_ckpt1['state_dict'], strict=False)
    resnet_34_model2.load_state_dict(resnet_34_ckpt2['state_dict'], strict=False)
    adapter15resnet_34_model2.load_state_dict(adapter15resnet_34_ckpt2['state_dict'], strict=False)

    image_dir1 = './n03100240/'
    image_dir2 = './n02797295'
    image_dir3 = './n04065272'
    result_dir1 = './n03100240_result'
    result_dir2 = './n02797295_result'
    result_dir3 = './n04065272_result'


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])

    # for file_name in os.listdir(image_dir1):
    #     pure_name = file_name.split('.')[0]
    #     file_path = os.path.join(image_dir1, file_name)
    #     save_path1 = os.path.join(result_dir1, pure_name + 'result_34.jpeg')
    #     save_path2 = os.path.join(result_dir1, pure_name + 'adapter15result_34.jpeg')
    #     print(save_path1)
    #     draw_CAM(resnet_34_model1, file_path, save_path1, image_transform, False)
    #     draw_CAM(adapter15resnet_34_model1, file_path, save_path2, image_transform, False)

    # for file_name in os.listdir(image_dir2):
    #     pure_name = file_name.split('.')[0]
    #     file_path = os.path.join(image_dir2, file_name)
    #     save_path1 = os.path.join(result_dir2, pure_name + 'result_34.jpeg')
    #     save_path2 = os.path.join(result_dir2, pure_name + 'adapter15result_34.jpeg')
    #     print(save_path1)
    #     draw_CAM(resnet_34_model2, file_path, save_path1, image_transform, False)
    #     draw_CAM(adapter15resnet_34_model2, file_path, save_path2, image_transform, False)

    for file_name in os.listdir(image_dir3):
        pure_name = file_name.split('.')[0]
        file_path = os.path.join(image_dir3, file_name)
        save_path1 = os.path.join(result_dir3, pure_name + 'result_34.jpeg')
        save_path2 = os.path.join(result_dir3, pure_name + 'adapter15result_34.jpeg')
        print(save_path1)
        draw_CAM(resnet_34_model2, file_path, save_path1, image_transform, False)
        draw_CAM(adapter15resnet_34_model2, file_path, save_path2, image_transform, False)




