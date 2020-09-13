#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
拒绝采样
'''

import numpy as np
import matplotlib.pyplot as plt


def cau(x):
    '''
    待采样分布（柯西分布）
    '''
    p = (1/np.pi)*(1/(1+x**2))
    return p


num = 3000
sample = []

for sam in range(num):
    x = np.random.uniform(low=-5, high=5)
    alpha = cau(x)/0.4

    u = np.random.uniform()
    if u < alpha:
        sample.append(x)

x = np.linspace(-5, 5, 100)
y = cau(x)

plt.plot(x, y, lw=2, label='cauchy distribution')
plt.hist(sample, bins=20, density=True, label='sample')

plt.legend()
plt.show()
