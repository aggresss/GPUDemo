#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
import argparse
from PIL import Image


# Traverse the picture files in a folder
def walk_dir(dir):
    image_list = []
    for root, _, files in os.walk(dir):
        for name in files:
            ext = os.path.splitext(name)[1][1:]
            if (ext.lower() == 'jpg'):
                path = root + os.sep + name
                image_list.append(path)
    return image_list


# save the resize picture
def resize_save(im, imgpath, width, length):
    if width == 0 and length == 0:
        return
    elif width != 0 or length != 0:
        size = auto_resize(im, width, length)
    else:
        size = (width, length)
    new_im = im.resize(size)
    new_im.save(imgpath)


# Calculate picture aspect ratio
def auto_resize(im, width, length):
    size = im.size
    if length == 0:
        length = int(float(width) / size[0] * size[1])
    if width == 0:
        width = int(float(length) / size[1] * size[0])
    return (int(width), int(length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='image-resize.py',
        usage=' python %(prog)s -p path -w wide -h height ',
        description='Batch modify image resolution.',
        epilog='SEE ALSO: http://github.com/aggresss/playground-python')

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default=None,
        help='file path you want to modify')
    parser.add_argument(
        '-w', '--width', type=int, default=0, help='modify width')
    parser.add_argument(
        '-l', '--length', type=int, default=0, help='modify height')
    argvs = parser.parse_args()
    if argvs.path is None:
        print('please input fielpath')
        sys.exit()
    # if argvs.path[-1] != os.path.sep:
    #     argvs.path = argvs.path + os.path.sep

    image_list = walk_dir(argvs.path)
    for imgpath in image_list:
        im = Image.open(imgpath)
        print(imgpath)
        resize_save(im, imgpath, argvs.width, argvs.length)