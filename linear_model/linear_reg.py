#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
线性回归，岭回归
'''

import numpy as np
import matplotlib.pyplot as plt


def data_generater(x):
    '''
    生成样本
    '''
    epsilon = np.random.normal(loc=0, scale=1.0, size=len(x))
    y = 3*x + 1 + 0.1*epsilon
    return y


def LinReg():
    '''
    线性回归
    y = \omega x + b
    y = \omega x_hat (omega = (omega ,b),x_hat=(x,1))
    L(w) = 1/2 ||x_hat * w_hat - y||^2
    '''
    x = np.linspace(0, 1, 500)
    y = data_generater(x)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_hat = np.c_[x, np.ones_like(x)]
    x_hatT = x_hat.T

    omega = np.linalg.inv(x_hatT.dot(x_hat))
    omega = omega.dot(x_hatT)
    omega = omega.dot(y)

    return omega, x, y


omega_hat, x, y = LinReg()
omega = omega_hat[0]
b = omega_hat[1]

test = np.linspace(0, 1, 78)
pred = omega * test + b

plt.plot(test, pred, lw=2, color='r', label='Linear Model')
plt.scatter(x, y, alpha=0.5, edgecolors='g', label='label')
plt.legend()
plt.title('linear regression')
plt.show()


def RidgeReg():
    '''
    岭回归
    y = \omega x + b
    y = \omega x_hat (omega = (omega ,b),x_hat=(x,1))
    L(w) = 1/2 ||x_hat * w_hat - y||^2 + 1/2 lambda *|| w_hat ||^2
    '''
    x = np.linspace(0, 1, 500)
    y = data_generater(x)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_hat = np.c_[x, np.ones_like(x)]
    x_hatT = x_hat.T

    lam = 1e-3

    xTx = x_hatT.dot(x_hat)
    I = np.eye(len(xTx))
    omega = np.linalg.inv(xTx+lam*I)
    omega = omega.dot(x_hatT)
    omega = omega.dot(y)

    return omega, x, y


omega_hat, x, y = LinReg()
omega = omega_hat[0]
b = omega_hat[1]

test = np.linspace(0, 1, 78)
pred = omega * test + b

plt.plot(test, pred, lw=2, color='r', label='Linear Model')
plt.scatter(x, y, alpha=0.5, edgecolors='g', label='label')
plt.legend()
plt.title('Ridge Regression')
plt.show()
