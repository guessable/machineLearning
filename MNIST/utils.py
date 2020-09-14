#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT

import matplotlib.pyplot as plt
import torch
import os
import shutil

def feature_image():
    pass


def save_checkpoints(state, is_best=None, base_dir='checkpoints', save_dir=''):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)
