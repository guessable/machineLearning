#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


data = load_iris()['data']
label = load_iris()['target']
x_train, x_test, y_train, y_test = train_test_split(data,
                                                    label,
                                                    test_size=0.2)

################### 逻辑回归 #################
log_reg = LogisticRegression(penalty='l2',
                             solver='newton-cg')
log_reg.fit(x_train, y_train)
score = log_reg.score(x_test, y_test)
print(score)

log_reg_cv = LogisticRegressionCV(penalty='l2',
                                  solver='newton-cg',
                                  cv=3)
log_reg_cv.fit(x_train, y_train)
score = log_reg_cv.score(x_test, y_test)
print(score)
