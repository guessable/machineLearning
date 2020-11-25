#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from baseModel import LinearModel


class LinearClassification(LinearModel):
    '''
    lam: L2
    alpha: L1
    basis: basis function
    '''

    def __init__(self, lam, alpha, basis=None):
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.basis = basis

    # TODO fit
    def fit(self, X, y):
        pass

    # TODO score
    def score(self, X, y):
        pass

    # TODO prediction
    def prediction(self, X):
        pass
