from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

file_path = '../images/car.JPEG'
img = Image.open(file_path)  # img为PIL Image类型的图片

transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.ToTensor(),
    ])



# transforms的使用
# 功能一：以tensor的方式打开图片，用到的函数是ToTensor()
img_tensor = transform_train(img)  # tensor_trans接收一个图片，可以是PIL Image类型也可以是numpy.array类型
img_tensor.save("./t3_image.jpg")

img_tensor.show()