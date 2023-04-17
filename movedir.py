# srcfile 需要复制、移动的文件
# dstpath 目的地址

import os
import shutil
from glob import glob

def mymovefile(srcfile, dstpath):  # 移动函数s
    shutil.move(srcfile, dstpath)
    print("move %s -> %s" % (srcfile, dstpath))

src_dir = './'
# dst_dir = './A/'  # 目的路径记得加斜杠
src_file_list = glob(src_dir + '*')  # glob获得路径下所有文件，可根据需要修改
print(src_file_list)
count = 0
for srcfile in src_file_list:
    if srcfile == './A' or srcfile == './B' or srcfile == './movedir.py':
        continue
    if count < 500:
        mymovefile(srcfile, './A/')
    elif count >= 500:
        mymovefile(srcfile, './B/')
    count += 1