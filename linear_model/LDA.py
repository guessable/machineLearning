#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
线性判别分析
'''

import numpy as np
import matplotlib.pyplot as plt


def class_one(x):
    '''
    类别1：
        产生于y=x(x+1)
    '''
    y = x*x + 1
    X = np.c_[x, y]
    return X


def class_two(x):
    '''
    类别2:
        产生于y=x-1
    '''
    y = -1 - x*x
    X = np.c_[x, y]
    return X


x = np.linspace(-2, 2)
X0 = class_one(x)
X1 = class_two(x)

mu_0 = X0.mean(axis=0)
mu_0 = mu_0.reshape(-1, 1)
mu_1 = X1.mean(axis=0)
mu_1 = mu_1.reshape(-1, 1)

sum0 = 0
for idx in range(len(X0)):
    X = X0[idx].reshape(-1, 1)
    sum0 += ((X-mu_0)).dot((X-mu_0).T)

sum1 = 0
for idx in range(len(X1)):
    X = X1[idx].reshape(-1, 1)
    sum1 += ((X-mu_1)).dot((X-mu_1).T)

Sw = sum0+sum1
W = np.linalg.inv(Sw).dot(mu_0-mu_1)

y0 = X0.dot(W)
y1 = X1.dot(W)
mu_0 = (mu_0.T).dot(W)
mu_1 = (mu_1.T).dot(W)

plt.scatter(X0[:, 0:1],
            X0[:, 1:2],
            label='class 0',
            alpha=0.5,
            edgecolors='g')
plt.scatter(X1[:, 0:1],
            X1[:, 1:2],
            label='class 1',
            alpha=0.5,
            edgecolors='c')
plt.scatter(y0, np.zeros_like(y0), label='projection ponits of class one')
plt.scatter(y1, np.zeros_like(y1), label='projection points of class two')

plt.scatter(mu_0, 0, marker='x', label=r'projection ponint of $\mu_0$')
plt.scatter(mu_1, 0, marker='x', label=r'projection ponint of $\mu_1$')

plt.legend()

plt.show()
