#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# Author        : Cheng Tao
# Last modified : 2020-09-10 20:09
# Filename      : tes.py

import numpy as np
import matplotlib.pyplot as plt


def Gauss(x):
    g = (1/(np.sqrt(2*np.pi))*np.exp(-1*(x-1)**2/(2)))
    return g


T = 10000
sample = [0. for i in range(T)]

for t in range(T-1):

    t = t + 1
    x_star = np.random.normal(loc=sample[t-1], scale=1.0)
    alpha = min(1, Gauss(x_star)/Gauss(sample[t-1]))

    u = np.random.uniform()

    if u < alpha:
        sample[t] = x_star
    else:
        sample[t] = sample[t-1]

x = np.linspace(-5, 5, 500)
y = Gauss(x)

plt.plot(x, y, lw=2, label='Gaussian distribution')
plt.hist(sample, bins=40, density=True, label='sample')
plt.legend()
plt.show()
