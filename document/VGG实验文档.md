### 一、模型train scrach

#### Vgg-16 train scarch

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/resnet_16_bn_cifar10 --arch vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

