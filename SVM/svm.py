#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
软间隔支持向量机
'''
import numpy as np
import matplotlib.pyplot as plt


class SVM():
    def __init__(self, train_data, train_label, C=200, epsilon=1e-2):
        '''
        train_data : (m,n) m样本数
        kernel : 核函数
        C : 惩罚系数
        epsilon : 松弛变量
        '''
        self.train_data = train_data
        self.train_label = train_label
        self.C = C
        self.epsilon = epsilon
        self.dim_Kmatrix = self.train_data.shape[0]
        self.k_matrix = self.kernel_matrix()
        self.alpha = np.array([0 for i in range(self.dim_Kmatrix)])
        self.b = 0

    def kernel(self, x, y):
        '''
        高斯核函数
        '''
        norm = (x-y)**2
        norm = norm.sum(axis=1)
        sigma = 10
        ker = np.exp(-1*norm/(2*sigma**2))
        return ker

    def kernel_matrix(self):
        '''
        计算核矩阵
        '''
        k_matrix = np.empty((self.dim_Kmatrix, self.dim_Kmatrix))
        for i in range(self.dim_Kmatrix):
            for j in range(i, self.dim_Kmatrix):
                k_ij = self.kernel(self.train_data[[i]],
                                   self.train_data[[j]])
                k_matrix[i][j] = k_ij
                k_matrix[j][i] = k_ij
        return k_matrix

    def f(self, idx):
        '''
        输出预测值
        idx : train_data 的索引
        '''
        alpha = self.alpha.reshape(1, -1)
        f = self.k_matrix[:, idx:idx+1]
        f = f*self.train_label
        f = alpha.dot(f)
        f = f + self.b
        return f

    def if_KKT(self, alpha, idx):
        '''
        判断alpha[idx]是否满足KKT条件
        '''
        yi_fi = self.train_label[idx] * self.f(idx)
        if np.abs(alpha[idx]) < self.epsilon:
            if yi_fi > 1 - self.epsilon:
                return True
            else:
                return False
        elif self.epsilon <= alpha[idx] and alpha[idx] < self.C-self.epsilon:
            if np.abs(yi_fi - 1) < self.epsilon:
                return True
            else:
                return False
        elif (self.C-self.epsilon) <= alpha[idx] and (self.C+self.epsilon) >= alpha[idx]:
            if yi_fi < 1 + self.epsilon:
                return True
            else:
                return False
        else:
            return False

    def update_b(self, E_i, E_j, i, j, alphai_old, alphaj_old):
        '''
        更新阀值b
        '''
        b_1 = -1*E_i - self.train_label[i]*self.k_matrix[i][i] * \
            (self.alpha[i]-alphai_old) - \
            self.train_label[j]*self.k_matrix[i][j] * \
            (self.alpha[j] - alphaj_old) + self.b
        b_2 = -1*E_j - self.train_label[j]*self.k_matrix[i][j] * \
            (self.alpha[j]-alphai_old) - \
            self.train_label[j]*self.k_matrix[j][j] * \
            (self.alpha[j] - alphaj_old) + self.b
        if self.alpha[i] > self.epsilon and self.alpha[i] < self.C-self.epsilon:
            self.b = b_1
        elif self.alpha[j] > self.epsilon and self.alpha[j] < self.C-self.epsilon:
            self.b = b_2
        else:
            self.b = (b_1+b_2)/2
        return self.b

    def smo(self):
        '''
        SMO算法
        外层循环选出不符合KKT条件的参数
        内层循环选出使得|E_i - E_j|最大的E_j
        '''
        for i in range(self.alpha.shape[0]):
            E_list = []
            if self.if_KKT(self.alpha, i) == False:
                E_i = self.f(i) - self.train_label[i]
                for j in range(self.dim_Kmatrix):
                    E_j = self.f(j) - self.train_label[j]
                    abs_Eij = np.abs(E_i - E_j)
                    if not E_list:
                        E_list = [abs_Eij]
                    else:
                        E_list.append(abs_Eij)
                idx_j = np.argmax(E_list)
                E_j = self.f(idx_j) - self.train_label[idx_j]
                eta = self.k_matrix[i][i] + \
                    self.k_matrix[idx_j][idx_j] - 2*self.k_matrix[i][idx_j]
                alpha_j = self.alpha[idx_j] + \
                    (self.train_label[idx_j]*(E_i-E_j))/eta
                alphai_old = self.alpha[i]
                alphaj_old = self.alpha[idx_j]
                if self.train_label[i] != self.train_label[j]:
                    L = max(0, self.alpha[idx_j]-self.alpha[i])
                    H = min(self.C, self.C+alphaj_old-alphai_old)
                else:
                    L = max(0, alphai_old+alphaj_old-self.C)
                    H = min(self.C, alphai_old+alphaj_old)
                if alpha_j > H:
                    alpha_j = H
                elif alpha_j <= H and alpha_j >= L:
                    pass
                else:
                    alpha_j = L
                alpha_i = self.alpha[i] + self.train_label[i] * \
                    self.train_label[idx_j] * (self.alpha[idx_j] - alpha_j)
                self.alpha[i] = alpha_i
                self.alpha[idx_j] = alpha_j
                self.b = self.update_b(E_i, E_j, i,
                                       idx_j, alphai_old, alphaj_old)
        return self.alpha, self.b

    def is_done(self):
        '''
        判断是否停止迭代
        '''
        if (self.alpha*self.train_label.reshape(-1)).sum() == 0:
            is_zero = True
        for i in range(self.alpha.shape[0]):
            y_i = self.train_label[i]
            if self.alpha[0] == 0 and self.f(i)*y_i >= 1:
                is_sta = True
            else:
                is_sta = False
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                if y_i*self.f(i) == 1:
                    is_sta1 = True
                else:
                    is_sta1 = False
            if self.alpha[i] == self.C and y_i*self.f(i) <= 1:
                is_sta2 = True
            else:
                is_sta2 = False
        done = is_sta and is_sta1 and is_sta2 and is_zero
        return done

    def train(self, epochs=1000):
        '''
        迭代求解
        '''
        epoch = 0
        done = False
        while(epoch < epochs and not done):
            epoch += 1
            alpha, b = self.smo()
            print(f'迭代次数:{epoch}')
            done = self.is_done()
        print(self.alpha)
        print((self.alpha*self.train_label.reshape(-1)).sum())
        return self.alpha, self.b

    def test(self, test_data, test_label):
        '''
        决策函数
        '''
        pred = 0
        test_label = test_label.reshape(-1)
        for i in range(self.dim_Kmatrix):
            pred += self.alpha[i] * \
                self.kernel(self.train_data[[i]], test_data) * \
                self.train_label[i]
        pred = pred + self.b
        pred = pred.reshape(-1)
        for i in range(len(pred)):
            if pred[i] >= 1-self.epsilon:
                pred[i] = 1
            else:
                pred[i] = -1
        idx = np.where(pred == test_label)
        exact = pred[idx]
        percent = exact.shape[0]/test_label.shape[0]
        return percent


if __name__ == '__main__':
    mean = [-3, -3]
    cov = [[4, 0],
           [0, 4]]
    x_1 = np.random.multivariate_normal(mean, cov, size=150)
    x_1 = np.c_[x_1, np.ones(len(x_1))]

    mean = [3, 3]
    cov = [[2, 0],
           [0, 2]]
    x_2 = np.random.multivariate_normal(mean, cov, size=150)
    x_2 = np.c_[x_2, -1*np.ones(len(x_2))]

    data = np.r_[x_1, x_2]
    np.random.shuffle(data)
    train_data = data[0:200, 0:2]
    train_label = data[0:200, 2:3]

    test_data = data[200:300, 0:2]
    test_label = data[200:300, 2:3]

    svm = SVM(train_data, train_label)
    svm.train()
    percent = svm.test(test_data, test_label)
    print(f'测试正确率:{percent*100}%')
