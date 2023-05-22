import cv2 as cv
import torch

def fun1():
    img = cv.imread('./images/flowers1.jpg')
    cv.imshow('img', img)
    cv.waitKey(-1)

def fun2():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    print(a * b)

if __name__ == '__main__':
    # fun1()
    fun2()