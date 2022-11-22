import utils

if __name__ == '__main__':
    # xlist = [0.6, 0.7, 0.8, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1]
    xlist = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]

    for item in xlist:

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.59cifar10tocifar10_resnet_56_pruned_{} --ci_dir ./calculated_ci/93.59_resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.59_resnet_56_cifar10.pth.tar --sparsity [0.]*29+[{}]*1'.format(str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.75cifar10tocifar10_resnet_56_pruned_{}_4 --ci_dir ./calculated_ci/93.75_resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 80 --lr_type cos --learning_rate 0.003 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.75_resnet_56_cifar10.pth.tar --sparsity [0.]*29+[{}]*1'.format(str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.75cifar10tocifar10_resnet_56_pruned_{}_5 --ci_dir ./calculated_ci/93.75_resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 80 --lr_type cos --learning_rate 0.003 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.75_resnet_56_cifar10.pth.tar --sparsity [0.]*28+[{}]*1+[0.]*1'.format(str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_56 --finetune_arch resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.59cifar10tocifar10_resnet_56_pruned_{}_2 --ci_dir ./calculated_ci/93.59_resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.59_resnet_56_cifar10.pth.tar --sparsity [0.]*28+[{}]*1+[0.]*1'.format(
        #     str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 --result_dir ' \
        #        './result/normal_pruned/68.70cifar100tocifar100_resnet_20_pruned_{}_2 --ci_dir ./calculated_ci/68.70_resnet_20_cifar100 ' \
        #        '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/68.70_resnet_20_cifar100.pth.tar --sparsity [0.]*8+[{}]+[0.]*3'.format(
        #     str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 --result_dir ' \
        #        './result/normal_pruned/68.70cifar100tocifar100_resnet_20_pruned_{}_4 --ci_dir ./calculated_ci/68.70_resnet_20_cifar100 ' \
        #        '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.001 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/68.70_resnet_20_cifar100.pth.tar --sparsity [0.]*7+[{}]+[0.]*4'.format(
        #     str(item), str(item))

        rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
               '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 --result_dir ' \
               './result/normal_pruned/68.70cifar100tocifar100_resnet_20_pruned_{}_6 --ci_dir ./calculated_ci/68.70_resnet_20_cifar100 ' \
               '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.002 --momentum 0.9 --weight_decay 0.0005 ' \
               '--graf --pretrain_dir ./pretrained_models/68.70_resnet_20_cifar100.pth.tar --sparsity [0.]*8+[{}]+[0.]*3'.format(
            str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 --result_dir ' \
        #        './result/normal_pruned/92.21cifar10tocifar10_resnet_20_pruned_{} --ci_dir ./calculated_ci/92.21_resnet_20_cifar10 ' \
        #        '--batch_size 128 --epochs 150 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/92.21_resnet_20_cifar10.pth.tar --sparsity [0.]*11+[{}]*1'.format(
        #     str(item), str(item))

        # rcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch resnet_20 --finetune_arch resnet_20 --result_dir ' \
        #        './result/normal_pruned/92.21cifar10tocifar10_resnet_20_pruned_{}_2 --ci_dir ./calculated_ci/92.21_resnet_20_cifar10 ' \
        #        '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/92.21_resnet_20_cifar10.pth.tar --sparsity [0.]*10+[{}]*1+[0.]*1'.format(
        #     str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.73cifar10tocifar10_adapter15resnet_56_pruned_{} --ci_dir ./calculated_ci/93.73_adapter15resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.73_adapter15resnet_56_cifar10.pth.tar --sparsity [0.]*100 ' \
        #          '--adapter_sparsity [{}]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.73cifar10tocifar10_adapter15resnet_56_pruned_{}_4 --ci_dir ./calculated_ci/93.73_adapter15resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 80 --lr_type cos --learning_rate 0.003 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.73_adapter15resnet_56_cifar10.pth.tar --sparsity [0.]*100 ' \
        #          '--adapter_sparsity [{}]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.73cifar10tocifar10_adapter15resnet_56_pruned_{}_5 --ci_dir ./calculated_ci/93.73_adapter15resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 80 --lr_type cos --learning_rate 0.003 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.73_adapter15resnet_56_cifar10.pth.tar --sparsity [0.]*28+[{}]*1+[0.]*1 ' \
        #          '--adapter_sparsity [0.]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #        '--finetune_data_dir ./data --pretrained_arch adapter15resnet_56 --finetune_arch adapter15resnet_56 --result_dir ' \
        #        './result/normal_pruned/93.73cifar10tocifar10_adapter15resnet_56_pruned_{}_2 --ci_dir ./calculated_ci/93.73_adapter15resnet_56_cifar10 ' \
        #        '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/93.73_adapter15resnet_56_cifar10.pth.tar --sparsity [0.]*28+[{}]*1+[0.]*1 ' \
        #          '--adapter_sparsity [0.]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #          '--finetune_data_dir ./data --pretrained_arch adapter15resnet_20 --finetune_arch adapter15resnet_20 --result_dir ' \
        #          './result/normal_pruned/92.22cifar10tocifar10_adapter15resnet_20_pruned_{} --ci_dir ./calculated_ci/92.22_adapter15resnet_20_cifar10 ' \
        #          '--batch_size 128 --epochs 150 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #          '--graf --pretrain_dir ./pretrained_models/92.22_adapter15resnet_20_cifar10.pth.tar --sparsity [0.]*100 ' \
        #          '--adapter_sparsity [{}]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar10 --finetune_dataset cifar10 ' \
        #          '--finetune_data_dir ./data --pretrained_arch adapter15resnet_20 --finetune_arch adapter15resnet_20 --result_dir ' \
        #          './result/normal_pruned/92.22cifar10tocifar10_adapter15resnet_20_pruned_{}_2 --ci_dir ./calculated_ci/92.22_adapter15resnet_20_cifar10 ' \
        #          '--batch_size 128 --epochs 200 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #          '--graf --pretrain_dir ./pretrained_models/92.22_adapter15resnet_20_cifar10.pth.tar --sparsity [0.]*10+[{}]*1+[0.]*1 ' \
        #          '--adapter_sparsity [0.]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
        #        '--finetune_data_dir ./data --pretrained_arch adapter19resnet_20 --finetune_arch adapter19resnet_20 --result_dir ' \
        #        './result/normal_pruned/68.72cifar100tocifar100adapter19_resnet_20_pruned_{}_2 --ci_dir ./calculated_ci/68.72_adapter19resnet_20_cifar100 ' \
        #        '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 ' \
        #        '--graf --pretrain_dir ./pretrained_models/68.72_adapter19resnet_20_cifar100.pth.tar --sparsity [0.]*12 ' \
        #        '--adapter_sparsity [{}]*1'.format(str(item), str(item))

        # adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
        #          '--finetune_data_dir ./data --pretrained_arch adapter19resnet_20 --finetune_arch adapter19resnet_20 --result_dir ' \
        #          './result/normal_pruned/68.72cifar100tocifar100adapter19_resnet_20_pruned_{}_4 --ci_dir ./calculated_ci/68.72_adapter19resnet_20_cifar100 ' \
        #          '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.001 --momentum 0.9 --weight_decay 0.0005 ' \
        #          '--graf --pretrain_dir ./pretrained_models/68.72_adapter19resnet_20_cifar100.pth.tar --sparsity [0.]*7+[{}]+[0.]*4 ' \
        #          '--adapter_sparsity [0.]*1'.format(str(item), str(item))

        adrcmd = 'python graf_prune_finetune_cifar_n.py --pretrained_dataset cifar100 --finetune_dataset cifar100 ' \
               '--finetune_data_dir ./data --pretrained_arch adapter19resnet_20 --finetune_arch adapter19resnet_20 --result_dir ' \
               './result/normal_pruned/68.72cifar100tocifar100adapter19_resnet_20_pruned_{}_6 --ci_dir ./calculated_ci/68.72_adapter19resnet_20_cifar100 ' \
               '--batch_size 128 --epochs 50 --lr_type cos --learning_rate 0.002 --momentum 0.9 --weight_decay 0.0005 ' \
               '--graf --pretrain_dir ./pretrained_models/68.72_adapter19resnet_20_cifar100.pth.tar --sparsity [0.]*12 ' \
               '--adapter_sparsity [{}]*1'.format(str(item), str(item))

        utils.execute_command([rcmd, adrcmd, adrcmd])
        # utils.execute_command([rcmd, rcmd, adrcmd])


