import utils

pruned_ratio = [10, 20, 30, 40, 50, 80, 90]
r20_pstr_list = ['[0.]+[0.05]*2+[0.05]*3+[0.1]*3+[0.1]*3', '[0.]+[0.1]*2+[0.1]*3+[0.1]*3+[0.2]*3', '[0.]+[0.1]*2+[0.2]*3+[0.2]*3+[0.3]*3',
                 '[0.]+[0.15]*2+[0.4]*3+[0.4]*3+[0.4]*3', '[0.]+[0.25]*2+[0.4]*3+[0.4]*3+[0.5]*3', '[0.]+[0.45]*2+[0.6]*3+[0.7]*3+[0.8]*3',
                 '[0.]+[0.5]*2+[0.7]*3+[0.8]*3+[0.9]*3']
adr20_pstr_list = ['[0.1]*1', '[0.1]*1', '[0.2]*1', '[0.4]*1', '[0.4]*1', '[0.7]*1', '[0.8]*1']

for i in range(len(pruned_ratio)):
    r20_pruned_cmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
                     '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 \
                    --result_dir ./result/normal_pruned/68.70_cifar100tocifar100_resnet_20_pruned_{} \
                    --ci_dir ./calculated_ci/68.70_resnet_20_cifar100 --batch_size 128 \
                    --epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
                    --graf --pretrain_dir ./pretrained_models/68.70_resnet_20_cifar100.pth.tar --sparsity {}'\
                    .format(str(pruned_ratio[i]), r20_pstr_list[i])
    cmd_list = [r20_pruned_cmd] * 3
    utils.execute_command(cmd_list)
    # cmd_list = []

    adr20_pruned_cmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
                       '--finetune_data_dir ./data --pretrained_arch adapter19resnet_20 --finetune_arch adapter19resnet_20 \
                        --result_dir ./result/normal_pruned/68.72_cifar100tocifar100_adapter19resnet_20_pruned_{} \
                        --ci_dir ./calculated_ci/68.72_adapter19resnet_20_cifar100 --batch_size 128 \
                        --epochs 300 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 \
                        --graf --pretrain_dir ./pretrained_models/68.72_adapter19resnet_20_cifar100.pth.tar --sparsity {}' \
                       ' --adapter_sparsity {}'.format(str(pruned_ratio[i]), r20_pstr_list[i], adr20_pstr_list[i])
    cmd_list = [adr20_pruned_cmd] * 3
    utils.execute_command(cmd_list)