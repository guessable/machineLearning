#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
高斯混合聚类
'''
import numpy as np
import matplotlib.pyplot as plt


def Gaussian(x, mu, cov):
    '''
    二维正态分布
    mu: 均值向量
    cov: 协方差矩阵
    '''
    px = (1/(2*np.pi*np.sqrt(np.linalg.det(cov)))) * \
        np.exp(-0.5*((x-mu).dot(np.linalg.inv(cov))).dot((x-mu).T))
    return px


def maxture(x, alpha, mu, cov):
    '''
    alpha: 系数向量
    mu: 矩阵，每行为均值向量
    cov: 三维tensor，第三维为协方差矩阵
    '''
    p1 = Gaussian(x, mu[[0]], cov[0])
    p2 = Gaussian(x, mu[[1]], cov[1])
    p3 = Gaussian(x, mu[[2]], cov[2])
    px = alpha[0]*p1 + alpha[1]*p2 + alpha[2]*p3
    return px


def Iter(x, alpha, mu, cov):
    '''
    x: 样本
    alpha: 系数向量
    mu: 矩阵，每行为均值向量
    cov: 三维tensor，第三维为协方差矩阵
    '''
    k = 3
    gamma = np.ones((len(x), k))
    for j in range(len(x)):
        for i in range(k):
            # 第j个样本为第i个正态分布生成的后验概率分布
            p = (alpha[i]*Gaussian(x[[j]], mu[[i]], cov[i])) / \
                maxture(x[[j]], alpha, mu, cov)
            gamma[j][i] = p
    # 更新参数
    for i in range(k):
        # mu
        mu_i = ((gamma[:, i:i+1]*x).sum(axis=0))/((gamma[:, i:i+1]).sum())
        # alpha
        alpha_i = (gamma[:, i:i+1]).mean()
        # cov
        cov_i = np.zeros((2, 2))
        for j in range(len(x)):
            cov_i += gamma[j][i]*((x[[j]]-mu[[i]]).T).dot(x[[j]]-mu[[i]])
        cov_i = cov_i/((gamma[:, i:i+1]).sum())

        mu[[i]] = mu_i
        alpha[i] = alpha_i
        cov[i] = cov_i
    return mu, alpha, cov


def MoGC(x, alpha, mu, cov, epochs):
    '''
    高斯混合聚类
    '''
    LLD = np.log(maxture(x, alpha, mu, cov)).sum()
    epsilon = 1
    epoch = 0
    while(not (epsilon < 1e-2) and epoch < epochs):
        LLD = np.log(maxture(x, alpha, mu, cov)).sum()

        mu, alpha, cov = Iter(x, alpha, mu, cov)
        next_LLD = np.log(maxture(x, alpha, mu, cov)).sum()

        epsilon = np.abs(next_LLD-LLD)
        epoch += 1
    return alpha, mu, cov


if __name__ == '__main__':
    epochs = 40
    mean = [3, -2]
    cov = [[4, 0.5], [0.5, 9]]
    sample = np.random.multivariate_normal(mean, cov, size=500)

    mu = sample[0:3, 0:3]
    cov = np.array([[[1.0, 0],
                     [0, 1.0]],
                    [[1.0, 0],
                     [0, 1.0]],
                    [[1.0, 0],
                     [0, 1.0]]])
    alpha = np.array([1/3, 1/3, 1/3])
    alpha, mu, cov = MoGC(sample, alpha, mu, cov, epochs)

    gamma = np.ones((len(sample), 3))
    for j in range(len(sample)):
        for i in range(3):
            p = (alpha[i]*Gaussian(sample[[j]], mu[[i]], cov[i])) / \
                maxture(sample[[j]], alpha, mu, cov)
            gamma[j][i] = p
    # 取gamma每行最大值索引
    max_idx = np.argmax(gamma, axis=1)

    idx_1 = np.where(max_idx == 0)
    C_1 = sample[idx_1]

    idx_2 = np.where(max_idx == 1)
    C_2 = sample[idx_2]

    idx_3 = np.where(max_idx == 2)
    C_3 = sample[idx_3]

    plt.figure()
    plt.subplot(121)
    plt.scatter(sample[:, 0:1],
                sample[:, 1:2],
                alpha=0.5)
    plt.title('sample')

    plt.subplot(122)
    plt.scatter(C_1[:, 0:1],
                C_1[:, 1:2],
                edgecolors='b',
                alpha=0.5)
    plt.scatter(C_2[:, 0:1],
                C_2[:, 1:2],
                edgecolors='y',
                alpha=0.5)
    plt.scatter(C_3[:, 0:1],
                C_3[:, 1:2],
                edgecolors='c',
                alpha=0.5)
    plt.title('Mixture-of-Gaussian clustering')
    plt.show()
