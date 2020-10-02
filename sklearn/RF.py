#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
随机森林：
    Bagging的特例，除行采样外
    还加入了随机列采样
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier


data = load_iris()['data']
label = load_iris()['target']

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    label,
                                                    test_size=0.1)

rf = RandomForestClassifier(n_estimators=50,
                            criterion='gini',
                            max_depth=8,
                            max_features='log2',  # 列采样
                            bootstrap=True)
rf.fit(x_train, y_train)
score = rf.score(x_test, y_test)
print(f'Classifier score:{score}')
