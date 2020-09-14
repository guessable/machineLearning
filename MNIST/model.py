#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader


class MNIST(nn.Module):
    """
    main block
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.init_weight(self.layer1[0])
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.init_weight(self.layer2[0])
        self.layer3 = nn.Sequential(nn.Linear(7*7*64, 1024),
                                    nn.ReLU())
        self.init_weight(self.layer3[0])
        self.layer4 = nn.Sequential(nn.Linear(1024, 10),
                                    # nn.Dropout(p=0.2),
                                    nn.Softmax())

    def __len__(self):
        return len(self.layer1)+len(self.layer2)+len(self.layer3)+len(self.layer4)

    def init_weight(self, layer):
        init.kaiming_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def view_feature(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        return out1[0, 0], out2[0, 0]

    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = out2.view(out2.size(0), -1)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out5


if __name__ == '__main__':
    model = MNIST()
    data = torch.randn(3, 1, 28, 28)
    out = model(data)
    print(out.sum(1))
    print(model)
    print(model.view_feature(data))
    print(len(model))
