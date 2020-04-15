#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'SmallWalnut'
__mtime__ = '2019/9/23'
# qq|wx:2456056533


"""
import os
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

from conf import Config
from createCode import create_verifycode_img
from log import klogger


class ImgDataset(data.Dataset):
    '''make 数据集'''
    transforms = transforms.Compose(
        transforms=[transforms.Resize(30),  # 这里压缩图片-压缩图片后计算会快很多很多，160*60 resize之后就是  80*30 ,等比缩小
                    # transforms.CenterCrop(30),  #  坑！！： Crop之后，就是 30*30  会导致丢掉特征
                    transforms.ToTensor(),  # img 转成 Tensor
                    # transforms.Normalize(
                    #     (0.1307,), (0.3081,)
                    # )

                    ])

    def __init__(self, config, transforms=transforms):

        self.img_dir = config.img_dir

        self.config = config
        self.char_len = config.CHAR_LEN
        self.char_num = config.CHAR_NUM
        self.transforms = transforms
        self.img_len = 0
        self.char26_dict = config._make_char26()
        self.samples = self.__get_img()

    def __len__(self):
        self.img_len = len(self.samples)
        return self.img_len

    def __getitem__(self, idx):
        img_path, img_label = self.samples[idx]

        img = Image.open(img_path).convert('L')
        img = self.config._img_resize(img)

        if self.transforms:
            img = self.transforms(img)

        return img, img_label

    def __get_img(self):
        samples = []
        for file in os.listdir(self.img_dir):
            file_path = os.path.join(self.img_dir, file)
            if os.path.isfile(file_path):
                file_h = file.rsplit('.', 1)[0]
                if '_' in file_h:
                    file_name = file_h.split('_')[0]  # label_index  ==> '1282'
                else:
                    file_name = file_h

                # one_hot_label_tensor = self._one_hot(file_name)
                one_hot_label_tensor = self._ont_hot_by_torch(file_name)
                samples.append((file_path, one_hot_label_tensor))  # 图片和标签组成元组拼接到数组，后续__getitem__通过索引一次拿一对

        return samples

    def _ont_hot_by_torch(self, file_label: str):
        '''通过torch来one-hot'''
        label = [i for i in file_label]

        # 26字母转码
        label = self.__char26(label)
        label = torch.from_numpy(np.array(label)).long().view(self.char_len, -1)  # label 转 Tensor
        one_hot_label_tensor = torch.zeros(self.char_len, self.char_num).scatter_(1, label, 1)
        return one_hot_label_tensor.view(-1, self.char_len * self.char_num)  # [-1,40]

    def __char26(self, label: list):
        _label = []
        for i in label:
            i = i.lower()
            char_idx = self.char26_dict.get(i, '')
            if char_idx:
                _label.append(int(char_idx))
            else:
                _label.append(int(i))

        return _label

    def _one_hot(self, file_label: str):
        '''弃用'''
        label = [i for i in file_label]
        # 26字母转码
        label = self.__char26(label)

        label_i = []
        for i in label:
            vec = [0] * self.char_num
            vec[i] = 1
            label_i += vec

        # list 转Tensor float()
        label_i_tensor = torch.from_numpy(np.array(label_i)).float()
        return label_i_tensor


class ImgModel(nn.Module):
    '''
    img_size: 160*80
    '''

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE', DEVICE)

    def __init__(self, config):
        super(ImgModel, self).__init__()
        self.config = config
        self.char_len = config.CHAR_LEN
        self.char_num = config.CHAR_NUM

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.per = config.per

        self.model_dir = config.model_dir
        self.model_name = '{}.pth'.format(str(config.code_type))

        # 26字母反转
        make_char26 = config._make_char26()
        self.idx_char_dict = dict([v, k] for k, v in make_char26.items())

        self.conv = nn.Sequential(nn.Conv2d(1, 30, 5, padding=1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(30), nn.ReLU(),

                                  nn.Conv2d(30, 60, 3, 1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(num_features=60),
                                  nn.ReLU(),

                                  )

        self.lf1 = nn.Linear(

            in_features=60 * 6 * 18, out_features=180)

        self.lf2 = nn.Linear(in_features=180, out_features=self.char_num * self.char_len)

    def forward(self, input):

        x = self.conv(input)
        x = x.view(-1, 60 * 6 * 18)

        x = self.lf1(x)  # [-1,40]
        x = self.lf2(x)

        return x

    def train(self, imgDataset):
        loss_fn = nn.MultiLabelSoftMarginLoss()
        if self.DEVICE != 'cpu':  # GPU
            self.to(self.DEVICE)
            loss_fn.to(self.DEVICE)

        loss_fn = nn.MultiLabelSoftMarginLoss()  # MultiLabelSoftMarginLoss  # CrossEntropyLoss
        optimizer = optim.Adam(params=self.parameters(), )
        acces = []
        epoch_flag = False

        train_dataloder = data.DataLoader(dataset=imgDataset, batch_size=self.batch_size, shuffle=True)

        train_dataloder_dataset_len = len(train_dataloder.dataset)

        if train_dataloder_dataset_len < self.batch_size:
            # 图片数量少于 批次batch_size
            batch_acc_tatol = self.char_len * train_dataloder_dataset_len
        else:
            # 图片数量 大于 批次batch_size
            batch_acc_tatol = self.char_len * self.batch_size

        for epoch in range(self.epochs):
            if epoch_flag: break
            for batch_idx, (img, label) in enumerate(
                    train_dataloder):  # batch_idx == len(train_dataloder.dataset) /batch_size

                if self.DEVICE != 'cpu':
                    img, label = img.to(self.DEVICE), label.to(self.DEVICE)

                optimizer.zero_grad()
                output = self(img)

                label = label.view_as(output)
                loss = loss_fn(output, label)

                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:  # 每x个批次统计一次
                    batch_acc = self._acc_count(output, label, batch_acc_tatol)  # 当前批次batch准确度x%
                    acces.append(batch_acc)
                    per = sum(acces) / (len(acces) * 100) * 100.  # 总准确度x%
                    if per > self.per:
                        epoch_flag = True
                        break

                    klogger.info(
                        'time:{},第{}次epoch,第{}batch:,loss_item:{},该batch内准确度:{}%,总准确度per:{}%'.format(datetime.now(),
                                                                                                     epoch,
                                                                                                     batch_idx,
                                                                                                     loss.item(),
                                                                                                     batch_acc, per))

        model_path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.state_dict(), model_path)

    def _acc_count(self, output, label, batch_acc_tatol):
        '''准确率统计'''

        output, label = output.view(-1, self.char_num), label.view(-1, self.char_num)

        output_dim, label_dim = output.argmax(dim=1), label.argmax(dim=1)  # 获取列最大 索引值 ,返回 一个 索引一维标量

        acc_history = []
        for i, j in zip(label_dim, output_dim):
            if torch.equal(i, j):
                acc_history.append(1)  # 相同
            else:
                acc_history.append(0)

        # 单个批次内准确度,既一个batch_size=3,有2张图正确，则 batch_acc = 2/3*100
        batch_acc = sum(acc_history) / batch_acc_tatol * 100.
        return batch_acc

    def _load_model(self):

        if self.DEVICE != 'cpu':
            self.to(self.DEVICE)

        model_path = os.path.join(self.model_dir, self.model_name)

        self.load_state_dict(state_dict=torch.load(model_path))

    def predict(self, img: Image):
        '''单图片预测
        img == Image.open(file).convert('L')
        '''

        self._load_model()  # 加载模型

        img = self.config._img_resize(img)
        img_tensor = ImgDataset.transforms(img)  # # torch.Size([1, 30, 80])
        img_tensor = img_tensor.reshape((-1, 1, 30, 80))  # ==> torch.Size([2,1, 30, 80]) 添加一个批次维度

        with torch.no_grad():
            if self.DEVICE != 'cpu':
                img_tensor = img_tensor.to(self.DEVICE)

            output = self(img_tensor)

            output = output.view(-1, self.char_num)

            output_dim = output.argmax(dim=1)

            output_dim = output_dim.view(-1, self.char_len)

            # tensor操作 GPU,CPU 都可以
            # 如果 tensor转np, 需要GPU_tensor先转CPU_tensor才可以转np
            if self.DEVICE != 'cpu':
                output_dim = output_dim.cpu()  # data from GPU to CPU

            output_dim_np = output_dim.numpy()

            for i in range(len(output_dim_np)):
                img_vs = ''
                for j in output_dim_np[i]:
                    jv = self.idx_char_dict.get(str(j), '')
                    img_v = jv if jv else str(j)
                    img_vs += img_v
            return img_vs

    def validate(self, imgDataset):
        '''数据集测试 评估准确度'''

        self._load_model()

        val_dataloder = data.DataLoader(dataset=imgDataset, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():  # 预测不需要grad
            for img, label in val_dataloder:

                if self.DEVICE != 'cpu':
                    img, label = img.to(self.DEVICE), label.to(self.DEVICE)

                # print(img.size())  # torch.Size([2, 1, 30, 80])  # DataLoader 加上了批次维度
                output = self(img)  # (批次,40)

                output, label = output.view(-1, self.char_num), label.view(-1, self.char_num)  # (批次*4,10)

                output_dim, label_dim = output.argmax(dim=1), label.argmax(dim=1)  # 获取行最大索引值,返回 一个一维标量，[批次*4,1]

                output_dim, label_dim = output_dim.view(-1, self.char_len), label_dim.view(-1, self.char_len)  # (批次，4)

                if self.DEVICE != 'cpu':
                    output_dim, label_dim = output_dim.cpu(), label_dim.cpu()

                output_dim_np, label_dim_np = output_dim.numpy(), label_dim.numpy()

                for i in range(len(output_dim_np)):
                    img_vs = ''
                    label_vs = ''
                    for j in output_dim_np[i]:
                        jv = self.idx_char_dict.get(str(j), '')
                        img_v = jv if jv else str(j)
                        img_vs += img_v
                    for k in label_dim_np[i]:
                        kv = self.idx_char_dict.get(str(k), '')
                        label_v = kv if kv else str(k)
                        label_vs += label_v
                    klogger.info('结果：{}  预测值/实际值：{}/{}'.format('正确' if img_vs == label_vs else '错误', img_vs, label_vs))

        klogger.info('validate done')


def run(train=False):
    if train:

        conf = Config(code_type=501, epochs=500, batch_size=512, per=96.)
        create_verifycode_img(100, conf)

        img_dataset = ImgDataset(config=conf)
        ImgModel(conf).train(imgDataset=img_dataset)
    else:
        # 预测用新测试数据集
        conf = Config(code_type=501, )
        conf.img_dir = os.path.join(conf._IMG_DIR, 'test_{}'.format(conf.code_type))
        if not os.path.exists(conf.img_dir):
            os.makedirs(conf.img_dir)

        create_verifycode_img(2, conf)

        img_dataset = ImgDataset(config=conf)

        ImgModel(conf).validate(img_dataset)


def run_predict():
    # === 单张图预测
    conf = Config(code_type=400, )
    img_model = ImgModel(conf)
    img = Image.open(r'D:\working\pyWorking\codeIdentify\imgs\400\2879_00.png')
    img = img.convert('L')
    vcode = img_model.predict(img)
    print(vcode)


if __name__ == '__main__':
    # run(0)
    run_predict()
