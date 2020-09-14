#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
密度聚类
'''
import random
import numpy as np
import matplotlib.pyplot as plt


def dist(x, y):
    '''
    距离函数
    '''
    d = np.sqrt((x[:, 0:1]-y[:, 0:1])**2 + (x[:, 1:2]-y[:, 1:2])**2)
    return d


def coreObj(sample, epsilon, MinPts):
    '''
    找出核心对象
    sample: 字典类型，key为0 - len(sample)-1
    '''
    core_object = {}
    for i in range(len(sample)):
        count = 0
        for j in range(len(sample)):
            d = dist(sample[f'{i}'], sample[f'{j}'])
            if d < epsilon:
                count += 1
        if count-1 > MinPts:
            core_object[f'{i}'] = sample[f'{i}']
    return core_object


def Iter(sample, core_object,  epsilon):
    '''
    smaple:样本（字典类型）
    core:核心对象

    return:
        类别
        去掉类别k的样本集
        去掉类别k的核心对象集合
    '''
    # 随机选择一个核心对象
    key = random.choice(list(core_object))
    core = core_object[key]

    C_k = {}
    for key in sample.keys():
        d = dist(sample[key], core)
        if d < epsilon:
            C_k[key] = sample[key]

    # 去掉相同元素
    set_CK = set(C_k)
    set_sample = set(sample)
    set_coreObj = set(core_object)

    _key = set_sample - set_CK
    _sample = {}
    for key in _key:
        _sample[key] = sample[key]

    _key = set_coreObj - set_CK
    _coreObject = {}
    for key in _key:
        _coreObject[key] = core_object[key]

    return C_k, _sample, _coreObject


def densityCluster(sample, max_k, epsilon, MinPts):
    '''
    return:
        所有类别集合的集合
    '''
    k = 0
    core_object = coreObj(sample,
                          epsilon,
                          MinPts)
    class_all = {}
    while(k < max_k and not(not core_object)):
        C_k, sample, core_object = Iter(sample,
                                        core_object,
                                        epsilon)
        class_all[f'{k}'] = C_k
        k += 1
    return class_all


def dictToNumpy(dict_data):
    '''
    将值为列表或numpy数组的字典转换为numpy数组
    '''
    num_array = np.array([[]])
    for value in dict_data.values():
        if num_array.size == 0:
            num_array = value
        else:
            num_array = np.r_[num_array, value]
    return num_array


if __name__ == '__main__':
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    sample_num = np.random.multivariate_normal(mean, cov, size=700)
    sample = {}
    for idx in range(len(sample_num)):
        sample[f'{idx}'] = sample_num[[idx]]
    epsilon = 0.7
    MinPts = 30
    max_k = 20
    class_all = densityCluster(sample, max_k, epsilon, MinPts)
    print(len(class_all))

    plt.figure()
    plt.subplot(121)
    plt.scatter(sample_num[:, 0:1],
                sample_num[:, 1:2],
                alpha=0.5)
    plt.title('sample')
    plt.subplot(122)
    for key in class_all.keys():
        C_k = class_all[key]
        C_k = dictToNumpy(C_k)
        plt.scatter(C_k[:, 0:1],
                    C_k[:, 1:2],
                    alpha=0.5)
    plt.title('density based clustering')
    plt.show()
