# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# just test
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""
# 裁剪60%
python graf_prune_finetune_cifar_n.py --pretrained_dataset svhn --finetune_dataset svhn --finetune_data_dir ./data/svhn --pretrained_arch resnet_32 --finetune_arch resnet_32 \
--result_dir ./result/normal_pruned/96.58_svhntosvhn_resnet_32_pruned_60 \
--ci_dir ./calculated_ci/96.58_resnet_32_svhn --batch_size 128 \
--epochs 300 --lr_type cos --learning_rate 0.05 --momentum 0.9 --weight_decay 0.0005 \
--graf --pretrain_dir ./pretrained_models/96.58_resnet_32_svhn.pth.tar --sparsity [0.]+[0.3]*2+[0.4]*5+[0.5]*5+[0.6]*5

python calculate_ci_n.py --arch resnet_32 --repeat 5 --num_layers 31 \
--feature_map_dir ../conv_feature_map/96.58_resnet_32_svhn_repeat5
"""
