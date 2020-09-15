#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
主成分分析
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pca(sample, d_sub):
    '''
    sample: d x m
    d：维度
    m：样本数
    d_sub: 子空间维度
    '''
    sample = sample.T
    cov = sample.dot(sample.T)
    eig, fea_vector = np.linalg.eig(cov)
    fea_vector = fea_vector.T
    W = fea_vector[:, 0:d_sub]
    sample_down = (W.T).dot(sample)
    return sample_down.T


if __name__ == '__main__':
    mean = [3, 2, -1]
    cov = [[1, 0, 0],
           [0, 4, 0],
           [0, 0, 9]]
    sample = np.random.multivariate_normal(mean, cov, size=400)
    d_sub = 2
    sample_down = pca(sample, d_sub)

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(sample[:, 0:1], sample[:, 1:2], sample[:, 2:3])

    ax2 = fig.add_subplot(212)
    ax2.scatter(sample_down[:, 0:1],
                sample_down[:, 1:2],
                alpha=0.4,
                edgecolors='g')
    plt.show()
