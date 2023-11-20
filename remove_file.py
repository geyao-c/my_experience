import os
import argparse

def argsget():
    parser = argparse.ArgumentParser("check")
    parser.add_argument('--dir', type=str, default='./data', help='path to dataset')
    args = parser.parse_args()
    return args

def remove_file():
    args = argsget()
    files = os.listdir(args.dir)
    for file in files:
        filepath = os.path.join(args.dir, file)
        if not os.path.isdir(filepath):
            os.remove(filepath)
            print(file)

if __name__ == '__main__':
    remove_file()