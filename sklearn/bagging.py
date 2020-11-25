#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
Bagging:
    基于自助采样训练多个基学习器
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


data = load_iris()['data']
label = load_iris()['target']

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    label,
                                                    test_size=0.1)

log_reg = LogisticRegression(solver='newton-cg')

bag = BaggingClassifier(base_estimator=log_reg,
                        n_estimators=20,
                        max_samples=0.9,
                        max_features=0.9)
bag.fit(x_train, y_train)
score = bag.score(x_test, y_test)
print(f'Classifier score:{score}')
