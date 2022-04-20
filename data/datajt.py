from PIL import Image
import torch
from torchvision import transforms

def fun1():
    img = Image.open('./image/flower1.jpg')
    transform_list = transforms.Compose([transforms.RandomResizedCrop(32),
                        transforms.RandomRotation(45),
                        transforms.RandomHorizontalFlip()])
    img = transform_list(img)
    img.show()

if __name__ == '__main__':
    fun1()