#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
EM算法
'''

import numpy as np
import matplotlib.pyplot as plt


class EM():
    '''
    高斯混合模型
    '''

    def __init__(self, mu, sigma, pi, size=400):
        '''
        混合分布的真实参数
            mu : [mu_1,mu_2,mu_3] 均值
            sigma : [sigma_1,sigma_2,sigma_3] 方差
            pi : [pi_1,pi_2,pi_3] 系数
        size : 样本数
        train_param : 保存训练后的参数
        '''
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.size = size
        self.sample = self.sampler(self.size)
        self.train_param = []

    def gaussian(self, x, mu, sigma):
        '''
        高斯函数
        '''
        gauss = (1/(np.sqrt(2*np.pi)*sigma)) * \
            np.exp(-1*((x-mu)**2)/(2*sigma**2))
        return gauss

    def mixture(self, x, mu, sigma, pi):
        '''
        混合分布
        '''
        mix = pi[0]*self.gaussian(x, mu[0], sigma[0]) + \
            pi[1]*self.gaussian(x, mu[1], sigma[1]) + \
            pi[2]*self.gaussian(x, mu[2], sigma[2])
        return mix

    def sampler(self, size):
        '''
        拒绝采样
        q(x) = 1/10 [-5,5]上的均匀分布
        '''
        sample = []
        max_sigma = max(1/(np.sqrt(2*np.pi)*self.sigma[0]),
                        1/(np.sqrt(2*np.pi)*self.sigma[1]),
                        1/(np.sqrt(2*np.pi)*self.sigma[2]))
        k = 10*max_sigma
        while(len(sample) <= size):
            x = np.random.uniform(low=-5, high=5)
            alpha = self.mixture(x, self.mu, self.sigma, self.pi)/k

            U = np.random.uniform()
            if U < alpha:
                if not sample:
                    sample = [x]
                else:
                    sample.append(x)
            else:
                pass
        sample = np.array(sample)
        return sample

    def posterior(self, mu, sigma, pi):
        '''
        后验概率分布
        mu,sigma,pi : 待更新的参数
        return:
            返回一个矩阵，每一行为一个样本的后验概率分布
        '''
        gamma = np.empty((len(self.sample), 3))
        for i in range(len(self.sample)):
            for k in range(3):
                gamma[i][k] = (pi[k]*self.gaussian(self.sample[i], mu[k],
                                                   sigma[k])) / self.mixture(self.sample[i], mu, sigma, pi)
        return gamma

    def em(self, mu, sigma, pi):
        '''
        mu,sigma,pi
            EM算法中迭代的参数
        '''
        gamma = self.posterior(mu, sigma, pi)
        for k in range(3):
            mu_k = (((gamma[:, k:k+1]).reshape(1, -1))*self.sample).sum()
            mu_k = mu_k / (gamma[:, k:k+1]).sum()

            sigma_k = (((self.sample-mu[k])**2) *
                       (gamma[:, k:k+1].reshape(1, -1))).sum()
            sigma_k = sigma_k / gamma[:, k:k+1].sum()
            sigma_k = np.sqrt(sigma_k)

            pi[k] = (gamma[:, k:k+1].sum())/len(self.sample)
            sigma[k] = sigma_k
            mu[k] = mu_k
        return mu, sigma, pi

    def train(self, mu, sigma, pi, iterations=600, epsilon=1e-3):
        '''
        训练:
            mu,sigma,pi为初始化值
        当大于迭代次数或变化小于epsilon时训练结束
        '''
        param_change = 1
        iteration = 0
        while(iteration < iterations):
            mu_temp = mu
            mu, sigma, pi = self.em(mu, sigma, pi)
            iteration += 1
            param_change = np.linalg.norm(mu-mu_temp)
            if (iteration+1) % 50 == 0:
                print(f'迭代轮数:{iteration+1}')
                print(f'均值:{mu}')
                print(f'标准差:{sigma}')
                print(f'系数:{pi}')
        self.train_param = [mu, sigma, pi]
        print(sigma)
        return self.train_param

    def test(self):
        x = np.linspace(-5, 5, 500)
        y = self.mixture(x, self.mu, self.sigma, self.pi)
        plt.plot(x, y, '-b', label='Exact', lw=2)

        pred = self.mixture(x,
                            self.train_param[0],
                            self.train_param[1],
                            self.train_param[2])
        plt.plot(x, pred, '--r', label='EM', lw=2)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    mu_exact = np.array([-2, 0.5, 2])
    sigma_exact = np.array([1, 0.5, 0.25])
    pi_exact = np.array([0.25, 0.5, 0.25])

    mu_init = np.array([0.2, -0.2, 0.1])
    sigma_init = np.array([0.5, 1, 1.5])
    pi_init = np.array([0.35, 0.25, 0.4])

    E_M = EM(mu_exact, sigma_exact, pi_exact, size=3800)
    E_M.train(mu_init, sigma_init, pi_init, iterations=400)

    E_M.test()
