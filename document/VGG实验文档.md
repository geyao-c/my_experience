### 一、train scrach

#### Vgg-16 train scarch

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/vgg_16_bn_cifar10 --arch vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

##### Cifar 100

```python
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/vgg_16_bn_cifar100 --arch vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar100
```

#### adapter-Vgg-16 

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_cifar10 --arch adapter_vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

##### Cifar 100

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_cifar100 --arch adapter_vgg_16_bn --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar100
```

#### adapter-Vgg-16-v2

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v2_cifar10 --arch adapter_vgg_16_bn_v2 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

##### Cifar 100

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v2_cifar100 --arch adapter_vgg_16_bn_v2 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar100
```

#### adapter-Vgg-16-v3

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v3_cifar10 --arch adapter_vgg_16_bn_v3 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

##### Cifar 100

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v3_cifar100 --arch adapter_vgg_16_bn_v3 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar100
```

#### adapter-Vgg-16-v4

##### Cifar 10

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v4_cifar10 --arch adapter_vgg_16_bn_v4 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar10
```

##### Cifar 100

```
python train_scrach.py --data_dir ./data --result_dir ./result/scrach_train/adapter_vgg_16_bn_v4_cifar100 --arch adapter_vgg_16_bn_v4 --batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0005 --dataset cifar100
```



### 二、feature map计算

#### Vgg-16 

##### Cifar 10

```python
python calculate_feature_maps_n.py --arch vgg_16_bn --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar
```

##### Cifar 100

```python
python calculate_feature_maps_n.py --arch vgg_16_bn --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar
```

#### adapter-Vgg-16

##### Cifar 10

```python
python calculate_feature_maps_n.py --arch adapter_vgg_16_bn --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar
```

##### Cifar 100

```python
python calculate_feature_maps_n.py --arch adapter_vgg_16_bn --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/74.22_adapter_vgg_16_bn_cifar100.pth.tar

python calculate_feature_maps_n.py --arch adapter_vgg_16_bn --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/74.09_adapter_vgg_16_bn_cifar100.pth.tar
```

#### adapter-Vgg-16-v4

##### Cifar 10

```python
python calculate_feature_maps_n.py --arch adapter_vgg_16_bn_v4 --dataset cifar10 --data_dir \
./data --pretrain_dir ./pretrained_models/93.91_adapter_vgg_16_bn_v4_cifar10.pth.tar
```

##### Cifar 100

```python
python calculate_feature_maps_n.py --arch adapter_vgg_16_bn_v4 --dataset cifar100 --data_dir \
./data --pretrain_dir ./pretrained_models/73.78_adapter_vgg_16_bn_v4_cifar100.pth.tar
```

### 

### 三、ci计算

#### Vgg-16

##### Cifar 10

```python
python calculate_ci_n.py --arch vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/93.96_vgg_16_bn_cifar10_repeat5
```

##### Cifar 100

```python
python calculate_ci_n.py --arch vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/73.88_vgg_16_bn_cifar100_repeat5
```

#### adapter-Vgg-16

##### Cifar 10

```python
python calculate_ci_n.py --arch adapter_vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/93.99_adapter_vgg_16_bn_cifar10_repeat5
```

##### Cifar 100

```python
python calculate_ci_n.py --arch adapter_vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/74.22_adapter_vgg_16_bn_cifar100_repeat5

python calculate_ci_n.py --arch adapter_vgg_16_bn --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/74.09_adapter_vgg_16_bn_cifar100_repeat5
```

#### adapter-Vgg-16-v4

##### Cifar 10

```python
python calculate_ci_n.py --arch adapter_vgg_16_bn_v4 --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/93.91_adapter_vgg_16_bn_v4_cifar10_repeat5
```

##### Cifar 100

```python
python calculate_ci_n.py --arch adapter_vgg_16_bn_v4 --repeat 5 --num_layers 12 \
--feature_map_dir ../conv_feature_map/73.78_adapter_vgg_16_bn_v4_cifar100_repeat5
```

### 

### 四、normal pruned

#### Vgg-16 

##### cifar10

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

##### 裁剪87%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/93.96_cifar10tocifar10_vgg_16_pruned_83 \
--ci_dir ./calculated_ci/93.96_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar --sparsity [0.45]*7+[0.78]*5

python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/93.96_cifar10tocifar10_vgg_16_pruned_83 \
--ci_dir ./calculated_ci/93.96_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar --sparsity [0.45]*7+[0.78]*5
```

##### cifar100

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/73.88_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.50]*5
```

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/73.88_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.75]*5
```

##### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/93.96_cifar10tocifar10_vgg_16_pruned_83 \
--ci_dir ./calculated_ci/93.96_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.96_vgg_16_bn_cifar10.pth.tar --sparsity [0.30]*7+[0.75]*5
```

##### 裁剪87%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/normal_pruned/73.88_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.45]*7+[0.78]*5
```



### adapter-Vgg-16 

##### Cifar10

###### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.21]*7+[0.75]*4+[0.76]
```

###### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.30]*7+[0.75]*5

# lr = 0.01
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.30]*7+[0.75]*5
```

###### **裁剪87%**

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.45]*7+[0.78]*5

# lr = 0.01
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.45]*7+[0.78]*5
```

##### Cifar100

###### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/74.22_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/74.22_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/74.22_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.75]*4+[0.76]

python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/74.09_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/74.09_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/74.09_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.75]*4+[0.76]

python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/74.09_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/74.09_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/74.09_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.5]*5
```

###### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/93.99_cifar10tocifar10_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/93.99_adapter_vgg_16_bn_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.99_adapter_vgg_16_bn_cifar10.pth.tar --sparsity [0.30]*7+[0.75]*4+[0.755]
```

###### **裁剪87%**

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/74.22_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/74.22_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/74.22_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.45]*7+[0.78]*5

python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/normal_pruned/74.09_cifar100tocifar100_vgg_16_pruned_81 \
--ci_dir ./calculated_ci/74.09_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/74.09_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.45]*7+[0.78]*5
```

### adapter-Vgg-16-v4

##### Cifar10

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn_v4 --finetune_arch adapter_vgg_16_bn_v4 \
--result_dir ./result/normal_pruned/93.91_cifar10tocifar10_vgg_16_v4_pruned_81 \
--ci_dir ./calculated_ci/93.91_adapter_vgg_16_bn_v4_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/93.91_adapter_vgg_16_bn_v4_cifar10.pth.tar --sparsity [0.21]*7+[0.75]*5
```

##### Cifar100

##### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn_v4 --finetune_arch adapter_vgg_16_bn_v4 \
--result_dir ./result/normal_pruned/73.78_cifar100tocifar100_vgg_16_v4_pruned_81 \
--ci_dir ./calculated_ci/73.78_adapter_vgg_16_bn_v4_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/73.78_adapter_vgg_16_bn_v4_cifar100.pth.tar --sparsity [0.21]*7+[0.75]*5
```

### 五、graft pruned

#### Vgg-16

##### cifar100 to cifar10

###### 裁剪81%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/graf_pruned/73.88_cifar100tocifar10_vgg_16_bn_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.21]*7+[0.75]*5
```

###### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/graf_pruned/73.88_cifar100tocifar10_vgg_16_bn_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.30]*7+[0.75]*5
```

###### 裁剪87%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch vgg_16_bn --finetune_arch vgg_16_bn \
--result_dir ./result/graf_pruned/73.88_cifar100tocifar10_vgg_16_bn_pruned_81 \
--ci_dir ./calculated_ci/73.88_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/73.88_vgg_16_bn_cifar100.pth.tar --sparsity [0.45]*7+[0.78]*5
```

#### Adapter-vgg-16

##### cifar100 to cifar10

###### 裁剪83%

```python
python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar10 --finetune_data_dir ./data --pretrained_arch adapter_vgg_16_bn --finetune_arch adapter_vgg_16_bn \
--result_dir ./result/graf_pruned/74.09_cifar100tocifar10_adapter_vgg_16_bn_pruned_81 \
--ci_dir ./calculated_ci/74.09_adapter_vgg_16_bn_cifar100 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/74.09_adapter_vgg_16_bn_cifar100.pth.tar --sparsity [0.30]*7+[0.75]*5
```

