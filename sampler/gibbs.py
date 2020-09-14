#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

PI = np.pi
mu_1 = 0
mu_2 = 0
sigma_1 = 1
sigma_2 = 1
pho = 0.5


def TwoD_gauss(x, y, mu_1, mu_2, sigma_1, sigma_2, pho):
    a = 1/(2*PI*sigma_1*sigma_2*np.sqrt(1-pho**2))
    b_1 = -1/(2*(1-pho**2))
    b_2 = ((x-mu_1)**2)/sigma_1**2
    b_3 = 2*pho*((x-mu_1)*(y-mu_2))/(sigma_1*sigma_2)
    b_4 = ((y-mu_2)**2)/(sigma_2**2)
    result = a*np.exp(b_1*(b_2-b_3+b_4))
    return result


def mu_sigma(y):
    '''
    p(x|y)
    '''
    mu_3 = mu_1 + pho*(sigma_1/sigma_2)*(y-mu_2)
    sigma_3 = sigma_1*np.sqrt(1-pho**2)
    return mu_3, sigma_3


def mu_sigma_(x):
    '''
    p(y|x)
    '''
    mu_4 = mu_2 + pho*(sigma_2/sigma_1)*(x-mu_1)
    sigma_4 = sigma_2*np.sqrt(1-pho**2)
    return mu_4, sigma_4


T = 10000
sample = np.random.rand(T, 2)

for t in range(T-1):
    t = t + 1

    # update x
    mu_3, sigma_3 = mu_sigma(sample[t-1][1])
    x_star = np.random.normal(loc=mu_3, scale=sigma_3)
    sample[t-1][0] = x_star

    # update y
    mu_4, sigma_4 = mu_sigma_(sample[t-1][0])
    y_star = np.random.normal(loc=mu_4, scale=sigma_4)
    sample[t-1][1] = y_star


mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
x = np.random.multivariate_normal(mean, cov, 10000)

plt.scatter(x[:, 0:1], x[:, 1:2], c='r', alpha=0.5, label='Reference')
plt.scatter(sample[:, 0:1], sample[:, 1:2], alpha=0.2, label='gibbs')

plt.legend()

plt.show()
