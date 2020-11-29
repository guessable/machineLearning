#!/usr/bin/env python3
# _*_ coding:utf-8 _*_


import numpy as np
import matplotlib.pyplot as plt

'''
均值函数：0
核函数：Squared Exponential Kernel(高斯核)
'''


class GP():
    def __init__(self, X: np.array, l: float):
        '''
        X:1D array
        '''
        self.X = X
        self.l = l

    def _squared_exponential_kernel(self, x_i: np.array, x_j: np.array):
        '''
        X = (x_1,x_2,...x_n)
        '''
        dist = np.linalg.norm(x_i-x_j)**2
        k = np.exp(-dist/(2*self.l**2))
        return k

    def _mean(self):
        return np.zeros_like(self.X)

    def func(self, plot=True):
        '''
        X: 1-D array
        '''
        cov_matrix = np.empty((len(self.X), len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                cov_matrix[i, j] = self._squared_exponential_kernel(self.X[i],
                                                                    self.X[j])
        mean = self._mean()
        data = np.random.multivariate_normal(mean,
                                             cov_matrix,
                                             size=50)

        if plot:
            plt.figure()
            for dim in range(50):
                plt.plot(self.X, data[dim])
            plt.show()

        return data


if __name__ == '__main__':
    X = np.linspace(-4, 4, 50)
    gp = GP(X, 1.2)
    data = gp.func()
