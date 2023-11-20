import os
import argparse
import shutil

def argsget():
    parser = argparse.ArgumentParser("check")
    parser.add_argument('--src_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--dst_dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--part', type=str, default='0.7', help='path to dataset')
    args = parser.parse_args()
    return args

def partion_files():
    args = argsget()
    src_dirs = os.listdir(args.src_dir)
    # dst_dirs = os.listdir(args.dst_dir)
    for src_dir_name in src_dirs:
        src_dirpath = os.path.join(args.src_dir, src_dir_name)
        if not os.path.isdir(src_dirpath):
            print('{} is not dir'.format(src_dirpath))
            exit(1)
        dst_dirpath = os.path.join(args.dst_dir, src_dir_name)
        if not os.path.exists(dst_dirpath):
            os.makedirs(dst_dirpath)
        image_names = os.listdir(src_dirpath)
        total = int(len(image_names) * float(args.part))
        count = 0
        for image in image_names:
            image_path = os.path.join(src_dirpath, image)
            if os.path.isdir(image_path):
                print('{} this is dir'.format(image_path))
                exit(1)
            # dst_image_path = os.path.join(dst_dirpath, image)
            print('copy {} --> {}'.format(image_path, dst_dirpath))
            shutil.copy(image_path, dst_dirpath)
            count += 1
            if count >= total:
                break

if __name__ == '__main__':
    partion_files()