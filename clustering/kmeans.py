#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
k-means
k = 3
样本由混合二维正态分布产生
'''

import numpy as np
import matplotlib.pyplot as plt


def dist(x, y):
    '''
    x,y \in R^2
    二范数
    '''
    d = np.sqrt((x[:, 0:1]-y[:, 0:1])**2 + (x[:, 1:2]-y[:, 1:2])**2)
    return d


def Iter(sample, mu):

    C_1 = np.array([[]])
    C_2 = np.array([[]])
    C_3 = np.array([[]])

    for idx in range(len(sample)):
        lam = np.array([[]])
        for i in range(len(mu)):
            d = dist(sample[[idx]], mu[[i]])
            if lam.size == 0:
                lam = np.array([[d]])
            else:
                lam = np.r_[lam, [[d]]]
        max_idx = np.argmax(lam)

        if max_idx == 0:
            # 判断C_i是否非空
            if C_1.size == 0:
                C_1 = sample[[idx]]
            else:
                C_1 = np.r_[C_1, sample[[idx]]]
        elif max_idx == 1:
            if C_2.size == 0:
                C_2 = sample[[idx]]
            else:
                C_2 = np.r_[C_2, sample[[idx]]]
        else:
            if C_3.size == 0:
                C_3 = sample[[idx]]
            else:
                C_3 = np.r_[C_3, sample[[idx]]]
    return C_1, C_2, C_3


def kmeans(sample, mu, epochs=50):
    C_1, C_2, C_3 = Iter(sample, mu)

    for epoch in range(epochs):
        mu_1 = C_1.mean(axis=0).reshape(1, -1)
        mu_2 = C_2.mean(axis=0).reshape(1, -1)
        mu_3 = C_3.mean(axis=0).reshape(1, -1)
        mu = np.r_[mu_1, mu_2, mu_3]

        C_1, C_2, C_3 = Iter(sample, mu)
    return C_1, C_2, C_3


if __name__ == '__main__':
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    sample_1 = np.random.multivariate_normal(mean, cov, size=700)

    mean = [1.2, 2.4]
    cov = [[2, 1], [1, 2]]
    sample_2 = np.random.multivariate_normal(mean, cov, size=700)

    sample = 0.5*(sample_1+sample_2)

    idx = [34, 247, 363]
    mu = sample[idx]

    plt.figure()
    plt.subplot(221)
    plt.scatter(sample[:, 0:1], sample[:, 1:2], alpha=0.5)
    plt.title('sample')

    plt.subplot(222)
    C_1, C_2, C_3 = kmeans(sample, mu, epochs=1)
    plt.scatter(C_1[:, 0:1], C_1[:, 1:2], edgecolors='g', alpha=0.5)
    plt.scatter(C_2[:, 0:1], C_2[:, 1:2], edgecolors='r', alpha=0.5)
    plt.scatter(C_3[:, 0:1], C_3[:, 1:2], edgecolors='y', alpha=0.5)
    plt.title('iter=1')

    plt.subplot(223)
    C_1, C_2, C_3 = kmeans(sample, mu, epochs=3)
    plt.scatter(C_1[:, 0:1], C_1[:, 1:2], edgecolors='g', alpha=0.5)
    plt.scatter(C_2[:, 0:1], C_2[:, 1:2], edgecolors='r', alpha=0.5)
    plt.scatter(C_3[:, 0:1], C_3[:, 1:2], edgecolors='y', alpha=0.5)
    plt.title('iter=3')

    plt.subplot(224)
    C_1, C_2, C_3 = kmeans(sample, mu, epochs=5)
    plt.scatter(C_1[:, 0:1], C_1[:, 1:2], edgecolors='g', alpha=0.5)
    plt.scatter(C_2[:, 0:1], C_2[:, 1:2], edgecolors='r', alpha=0.5)
    plt.scatter(C_3[:, 0:1], C_3[:, 1:2], edgecolors='y', alpha=0.5)
    plt.title('iter=5')

    plt.show()
