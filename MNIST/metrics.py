#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT

import torch


class MetricTracker():
    def __init__(self):
        self.acc = 0
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.total = 0
        self.right_num = 0

    def update_avg(self, loss):
        self.count += 1
        self.sum += loss
        self.avg = self.sum/self.count

    def update_acc(self, out, lab):
        pred_num = torch.topk(out, 1)[1]
        lab_num = torch.topk(lab, 1)[1]
        right = torch.where(pred_num == lab_num, torch.ones_like(
            lab_num), torch.zeros_like(lab_num))
        right_num = right.sum()
        self.right_num += right_num.item()
        self.total += right.size(0)
        self.acc = self.right_num/self.total
        


if __name__ == '__main__':
    one_hot = torch.tensor([[0.0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0.0,1,0,0,0,0,0,0,0,0]])
    out = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0])
    print(torch.topk(one_hot, 1)[1])
