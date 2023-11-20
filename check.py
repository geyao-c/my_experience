import os
import argparse

def argsget():
    parser = argparse.ArgumentParser("check")
    parser.add_argument('--check_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--train_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--val_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--src_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dst_dir', type=str, default='./data', help='path to dataset')
    args = parser.parse_args()
    return args

def checknum1():
    args = argsget()
    files = os.listdir(args.check_dir)
    dir_count = 0
    for file in files:
        filepath = os.path.join(args.check_dir, file)
        # print(file)
        if os.path.isdir(filepath):
            dir_count += 1
            print(file)
    print(dir_count)

def check_consis():
    args = argsget()
    train_files = os.listdir(args.train_dir)
    val_files = os.listdir(args.val_dir)
    dir_count = 0
    for train_file in train_files:
        train_filepath = os.path.join(args.train_dir, train_file)
        if os.path.isdir(train_filepath):
            dir_count += 1
            if train_file not in val_files:
                print('{} not in'.format(train_file))
    print(dir_count)

def check_nums():
    args = argsget()
    src_dirs = os.listdir(args.src_dir)
    dst_dirs = os.listdir(args.dst_dir)
    src_files_count = 0
    dst_files_count = 0
    for dir in src_dirs:
        src_dirpath = os.path.join(args.src_dir, dir)
        if os.path.isdir(src_dirpath):
            src_files_count += len(os.listdir(src_dirpath))
    for dir in dst_dirs:
        dst_dirpath = os.path.join(args.dst_dir, dir)
        if os.path.isdir(dst_dirpath):
            dst_files_count += len(os.listdir(dst_dirpath))
    print(src_files_count)
    print(dst_files_count)

if __name__ == '__main__':
    # checknum1()
    # check_consis()
    check_nums()