#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'SmallWalnut'
__mtime__ = '2019/9/23'
# qq|wx:2456056533


"""

import os
from PIL import Image

L4_INT = 400  # 4位纯数字
L4_CHAR = 401  # 4位纯字母
L4_INT_CHAR = 402  # 4位 数字+字母组合
L5_INT = 500  # 5位纯数字
L5_CHAR = 501
L5_INT_CHAR = 502
L6_INT = 600
L6_CHAR = 601
L6_INT_CHAR = 602


class Config:
    '''
    配置类
    '''
    IMAGE_SIZE = (160, 60)  # 画长宽，可随意，后会等比转成resize成 _IMAGE_SIZE = (60, 60)

    _IMAGE_SIZE = (160, 60)  # 模型训练长宽，改其需改模型Linear层的输入

    BATCH_SIZE = 512  # 训练批次，一次搞几张图片，只有一张的话 批次就等于1

    root_dir = os.path.abspath(os.path.dirname(__file__))

    _IMG_DIR = os.path.join(root_dir, 'imgs')
    _MODEL_DIR = os.path.join(root_dir, 'models')

    def __init__(self, code_type, epochs=500, batch_size=512, per=96.):
        self.code_type = code_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.per = per

        self.img_dir = os.path.join(self._IMG_DIR, str(self.code_type))
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.model_dir = os.path.join(self._MODEL_DIR, str(self.code_type))

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if code_type == L4_INT:
            self.CHAR_LEN = 4
            self.CHAR_NUM = 10
            self.CHAR_SET = '0123456789'


        elif code_type == L4_CHAR:
            self.CHAR_LEN = 4
            self.CHAR_NUM = 26
            self.CHAR_SET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        elif code_type == L4_INT_CHAR:
            self.CHAR_LEN = 4
            self.CHAR_NUM = 10 + 26
            self.CHAR_SET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

        elif code_type == L5_INT:
            self.CHAR_LEN = 5
            self.CHAR_NUM = 10
            self.CHAR_SET = '0123456789'

        elif code_type == L5_CHAR:
            self.CHAR_LEN = 5
            self.CHAR_NUM = 26
            self.CHAR_SET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        elif code_type == L5_INT_CHAR:
            self.CHAR_LEN = 5
            self.CHAR_NUM = 10 + 26
            self.CHAR_SET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

        elif code_type == L6_INT:
            self.CHAR_LEN = 6
            self.CHAR_NUM = 10
            self.CHAR_SET = '0123456789'


        elif code_type == L6_CHAR:
            self.CHAR_LEN = 6
            self.CHAR_NUM = 26
            self.CHAR_SET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        elif code_type == L6_INT_CHAR:
            self.CHAR_LEN = 6
            self.CHAR_NUM = 10 + 26
            self.CHAR_SET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        else:
            assert 'code_type do not support now'

    def _img_resize(self, img: Image):
        '''不同尺寸的图片 等比 转换成 160*60

        '''
        img_size = img.size
        if img_size != self._IMAGE_SIZE:
            img = img.resize(self._IMAGE_SIZE, Image.ANTIALIAS)

        return img

    def _make_char26(self):
        char26 = self.CHAR_SET
        char26_dict = {}
        char26_index = 0
        for c in char26:
            char26_dict[c] = str(char26_index)  # str为后续好反转
            char26_index += 1
        return char26_dict
