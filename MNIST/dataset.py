#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT

import struct
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image
from model import MNIST

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(
            fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    labels = torch.from_numpy(labels)
    labels = torch.tensor(labels, dtype=torch.int64)
    return labels


class RoadData(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        onehot = F.one_hot(label, num_classes=10)
        image = TF.to_tensor(image)
        image = image.float()
        sample = {'img': image, 'lab': onehot}
        return sample


if __name__ == '__main__':
    image = decode_idx3_ubyte('./train-images.idx3-ubyte')
    label = decode_idx1_ubyte('./train-labels.idx1-ubyte')
    # pil = ToPILImage()
    # img = pil(image[100])
    # plt.imshow(img)
    # plt.show()
    # print(label[100])
    dataset=RoadData(image,label)
    # dataloader = DataLoader(dataset,batch_size=5,shuffle=True)
    # for idx,sample in enumerate(dataloader):
        # print(sample['img'].size())

    img=dataset[563]
    model = MNIST()
    model_para=torch.load('./checkpoints/MNIST/checkpoint.pth.tar')
    model.load_state_dict(model_para['model'])
    image = img['img'].reshape(1,1,28,28)
    lab = img['lab']
    pred = model(image)
    pred = torch.topk(pred,1)[1]
    true = torch.topk(lab,1)[1]
    print(f'pred:{pred}',
          f'true:{true}')
