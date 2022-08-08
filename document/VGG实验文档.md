### 一、train scrach

#### Vgg-16 train scarch

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/vgg_16_bn_cifar10 --arch vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

#### adapter-Vgg-16 

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_cifar10 --arch adapter_vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

### 二、feature map计算

#### Vgg-16 

##### Cifar 10

```python
python calculate_feature_maps_n.py --arch vgg_16_bn --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar
```

#### adapter-Vgg-16

##### Cifar 10

```python
python calculate_feature_maps_n.py --arch adapter_vgg_16_bn --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar
```

### 三、ci计算

#### Vgg-16

##### Cifar 10

```python
python calculate_ci_n.py --arch vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/93.96_vgg_16_bn_cifar10_repeat5
```

#### adapter-Vgg-16

##### Cifar 10

```python
python calculate_ci_n.py --arch adapter_vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/93.99_adapter_vgg_16_bn_cifar10_repeat5
```

### 四、normal pruned

#### Vgg-16 

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/93.96_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.96_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar --sparsity [0.21]*7+[0.75]*5
```

##### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/93.96_cifar10tocifar10_vgg_16_pruned_83 \
--ci_dir ./calculated_ci/93.96_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar --sparsity [0.30]*7+[0.75]*5
```

### adapter-Vgg-16 

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.21]*7+[0.75]*5
```

##### 