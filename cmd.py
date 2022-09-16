python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar100 --finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 \
--result_dir ./result/graf_pruned/93.59cifar10tocifar100_resnet_56_pruned_48 \
--ci_dir ./calculated_ci/93.59_resnet_56_cifar10 --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--pretrain_dir ./pretrained_models/93.59_resnet_56_cifar10.pth.tar --sparsity [0.]+[0.15]*2+[0.4]*9+[0.4]*9+[0.4]*9