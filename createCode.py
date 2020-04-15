#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'SmallWalnut'
__mtime__ = '2019/9/23'
# qq|wx:2456056533


"""
import os
import random
from multiprocessing import Process

from PIL import Image, ImageDraw, ImageFont
from captcha.image import ImageCaptcha

from conf import Config


# 通过 captcha 生成验证码数据集
def create_verifycode_img(img_num=10, config=''):
    image = ImageCaptcha(width=config.IMAGE_SIZE[0], height=config.IMAGE_SIZE[1])
    for i in range(img_num):
        label = ''.join(random.sample(config.CHAR_SET, config.CHAR_LEN))
        image.write(label, os.path.join(config.img_dir, label + '_' + str(i) + str(random.randrange(img_num)) + '.png'))


class ImgCoder:
    '''自己画验证码图片作为训练数据集'''

    font_dir = './fonts/'  # 默认字体
    size_cha = 0.8  # 字体大小调整(字体粗细,太大会导致画出异常)
    # 干扰物
    overlaps = [0.1, 0.2, 0.3]  # 重叠边距
    noises = ['', 'point', 'line', 'circular']
    # noises = ['point', 'line']

    bool_text_rotates = True  # 字体是否旋转
    bool_rd_text_overlaps = True  # 字体是否重叠
    bool_rd_text_pos = True  # 随机字符位置
    bool_rd_text_size = True  # 随机字体大小
    bool_rd_text_color = True  # 随机字体颜色，默认白纸黑字
    bool_rd_bg_color = True  # 随机背景颜色,默认白

    bool_rd_font_type = True  # 随机字体风格,默认字体目录第一个

    # bool_text_rotates = False  # 字体是否旋转
    # bool_rd_text_overlaps = False  # 字体是否重叠
    # bool_rd_text_pos = False  # 随机字符位置
    # bool_rd_text_size = False  # 随机字体大小
    # bool_rd_text_color = False  # 随机字体颜色，默认白纸黑字
    # bool_rd_bg_color = False  # 随机背景颜色,默认白
    #
    # bool_rd_font_type = False  # 随机字体风格,默认字体目录第一个

    randRGB_start = 0
    randRGB_end = 255

    def __init__(self, config):
        self.fonts = self.randFont()
        self.width_im, self.height_im = config.IMAGE_SIZE
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.CHAR_LEN = config.CHAR_LEN
        self.CHAR_SET = config.CHAR_SET

        self.img_dir = config.img_dir

    def randRGB(self):
        ''' random_bgcolor
        :return RGB
        '''
        return random.randint(self.randRGB_start, self.randRGB_end), random.randint(self.randRGB_start,
                                                                                    self.randRGB_end), random.randint(
            self.randRGB_start, self.randRGB_end)

    def randFont(self):
        font_paths = []
        for dirpath, dirnames, filenames in os.walk(self.font_dir):
            for filename in filenames:
                filepath = dirpath + os.sep + filename
                font_paths.append(filepath)
        return font_paths

    def captcha_draw(self, font_path=None, bg_color=(), text_color=()):
        '''

        :param font_path: 指定字体风格
        :param bg_color:   tuple(255,255,255)   背景，默认白色
        :param text_color: tuple(0,0,0)     字体，默认黑
        :return:
        '''
        derx = 0
        dery = 0

        if self.bool_rd_text_overlaps:
            overlap = random.choice(self.overlaps)  # 重叠程度
        else:
            overlap = 0.0

        width_cha = int(self.width_im / max(self.CHAR_LEN - overlap, 3))  # 字符区域宽度

        height_cha = self.height_im * 1.2  # 字符区域高度

        size_cha = self.size_cha

        if self.bool_rd_text_size:
            # 随机字符大小
            size_cha = random.uniform(self.size_cha - 0.1, self.size_cha + 0.1)

        char_size = int(size_cha * min(width_cha, height_cha) * 1.5)  # 字符大小

        if self.bool_rd_bg_color:
            _bg_color = self.randRGB() if not bg_color else bg_color  # 指定大于随机
        else:
            _bg_color = 'white' if not bg_color else bg_color

        im = Image.new(mode='RGB', size=self.IMAGE_SIZE, color=_bg_color)  # color 背景颜色，size 图片大小

        drawer = ImageDraw.Draw(im)
        contents = []

        for i in range(self.CHAR_LEN):
            if self.bool_rd_text_color:
                _text_color = self.randRGB() if not text_color else text_color

            else:
                _text_color = 'black' if not text_color else text_color

            if self.bool_rd_text_pos:
                derx = random.randint(0, int(max(width_cha - char_size - 5, 1)))
                dery = random.randint(0, int(max(height_cha - char_size - 5, 1)))

            char = random.choice(self.CHAR_SET)

            if self.bool_rd_font_type:
                font = random.choice(self.fonts) if not font_path else font_path
            else:
                font = self.fonts[0] if not font_path else font_path

            font_type = ImageFont.truetype(font, char_size)
            contents.append(char)
            im_cha = self.cha_draw(char, _text_color, font_type, self.bool_text_rotates, char_size)

            # 字符黏贴位置
            # print(width_cha)
            # print(height_cha)
            box_x = int(max(i - overlap, 0) * width_cha) + derx + 2
            box_y = int(dery + random.randint(1, 4))

            box = (box_x, box_y)
            # 字符黏贴
            im.paste(im_cha, box, im_cha)

        return im, drawer, contents

    def save_img(self, index, font_path=None, bg_color=(), text_color=()):
        im, drawer, contents = self.captcha_draw(font_path, bg_color, text_color)

        # 画干扰物
        self.baseDraw(drawer)

        #  save img
        img_name = ''.join(contents) + '_' + str(index) + str(
            random.randrange(10000)) + '.jpg'  # 0001_0.jpg  : 避免相同的文件名不同的干扰线导致同名只保留一个文件
        img_path = os.path.join(self.img_dir, img_name)
        im.save(img_path)

    def baseDraw(self, drawer):
        '''
        基本的画干扰物
        :return:
        '''
        if 'point' in self.noises:
            nb_point = 20
            color_point = self.randRGB()
            for i in range(nb_point):
                x = random.randint(0, self.width_im)
                y = random.randint(0, self.height_im)
                drawer.point(xy=(x, y), fill=color_point)
        if 'line' in self.noises:
            nb_line = 3
            for i in range(nb_line):
                color_line = self.randRGB()
                sx = random.randint(0, self.width_im)
                sy = random.randint(0, self.height_im)
                ex = random.randint(0, self.width_im)
                ey = random.randint(0, self.height_im)
                drawer.line(xy=(sx, sy, ex, ey), fill=color_line)

        if 'circular' in self.noises:
            nb_circle = 20
            color_circle = self.randRGB()
            for i in range(nb_circle):
                sx = random.randint(0, self.width_im - 10)
                sy = random.randint(0, self.height_im - 10)
                temp = random.randint(1, 5)
                ex = sx + temp
                ey = sy + temp
                drawer.arc((sx, sy, ex, ey), 0, 360, fill=color_circle)

    def cha_draw(self, char, text_color, font, bool_text_rotates, char_size, max_angle=40):
        '''
        画单个字符,后续黏贴
        :param char: 字符
        :param text_color:
        :param font:
        :param bool_text_rotates:
        :param char_size: 字符大小
        :param max_angle:  最大旋转角度
        :return:
        '''

        im_size = (char_size * 2, char_size * 2)
        if not im_size[0] < self.width_im and not im_size[1] < self.height_im:
            im_size = (int(char_size * 0.8), int(char_size * 0.8))

        im = Image.new(mode='RGBA', size=im_size)
        drawer = ImageDraw.Draw(im)
        drawer.text(xy=(0, 0), text=char, fill=text_color, font=font)  # text 内容，fill 颜色， font 字体
        if bool_text_rotates:
            angle = random.randint(-max_angle, max_angle)
            im = im.rotate(angle, Image.BILINEAR, expand=1)
        im = im.crop(im.getbbox())
        return im

    def run(self, img_num=1, font_path='./fonts/times.ttf', bg_color=None, text_color=None):

        for index in range(img_num):
            self.save_img(index, font_path, bg_color, text_color)


class WxImgCoder(ImgCoder):
    '''
     搜狗微信图片验证码生成
    '''

    IMAGE_SIZE = (203, 66)  # 验证码图片大小设置

    noises = ['line', 'point']
    bool_text_rotates = True
    bool_rd_text_size = True

    randRGB_start = 30
    randRGB_end = 120

    def baseDraw(self, drawer):
        '''波浪线'''
        pass

    def run(self, img_num=1, font_path='./fonts/times.ttf', bg_color=(165, 165, 165),
            text_color=None):  # bg_color=(165, 165, 165)
        for index in range(img_num):
            self.save_img(index, font_path, bg_color=bg_color, text_color=text_color)


if __name__ == '__main__':
    config = Config(code_type=400)

    # ImgCoder(config).run(img_num=1)
    # WxImgCoder(config).run()

    # create_verifycode_img(1, config)

    for i in range(2):
        # Process(target=ImgCoder(config).run(img_num=1)).start()
        Process(target=create_verifycode_img, args=(5, config)).start()
