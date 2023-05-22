import torch
from models.adapter_resnet_tinyimagenet import adapter3resnet_tinyimagenet_56

def fun1():
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    print(mean)
    print(std)

def fun2():
    # x = torch.size()
    pass

def fun3():
    model = adapter3resnet_tinyimagenet_56([0.]*100, 200, [0.]*100)
    print(model)

def fun4():
    x = 'a'
    y = 'b'
    # print(x[0] - y[0])
    x = 'dnisad'
    # x[0] = '0'
    for item in x:
        print(item)

    print(x)
if __name__ == '__main__':
    # fun1()
    # fun3()
    fun4()