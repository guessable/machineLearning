#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import graphviz

from xgboost import plot_tree
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


data = load_boston()['data']
label = load_boston()['target']

data = np.delete(data, [1, 2, 3, 8, 9], axis=1)


x_train, x_test, y_train, y_test = \
    train_test_split(data, label, test_size=0.1)


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

############################### origin API ##########################
param = {
    # General Parameters
    'booster': 'gbtree',  # booster
    'verbosity': 1,  # 是否打印训练信息
    'nthread': 4,  # 线程数

    # Learning Task Parameters
    'objective': 'reg:squarederror',  # 损失函数
    'eval_metric': 'rmse',  # 验证集评价方法

    # params for tree booster
    'eta': 0.1,
    'gamma': 0.1,  # 确定是否分裂的阀值
    'max_depth': 10,
    'min_child_weight': 2,
    'subsample': 0.8,  # 每次学习从数据集采样,行采样
    'colsample_bytree': 1,  # 列采样
    'sampling_method': 'uniform',
    'lambda': 2,  # L2正则化系数
    'alpha': 1,  # L1正则化系数
    'tree_method': 'auto'  # 分裂算法
}

num_boost_round = 1000
watchlist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param,
                dtrain,
                num_boost_round,
                evals=watchlist,
                early_stopping_rounds=20)
# save model
bst.save_model('0001.model')

# load model
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('0001.model')

# pred
pred = bst.predict(dtest)

# score
y_true_mean = y_test.mean()
v = ((y_test-y_true_mean)**2).sum()
u = ((y_test-pred)**2).sum()
score = (1-u/v)
print(f'score:{score}')

# feature importance
importance = bst.get_score()
print(importance)

# plot
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=1)
xgb.to_graphviz(bst, num_trees=2)
plt.show()

################################ 多分类 ######################
data = load_iris()['data']
label = load_iris()['target']

x_train, x_test, y_train, y_test = \
    train_test_split(data, label, test_size=0.1)


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

param = {
    'booster': 'gbtree',
    'verbosity': 0,
    'nthread': 4,
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'num_class': 3,  # 和mutil:softmax 配合使用
    'eta': 0.1,
    'gamma': 0.1,
    'max_depth': 7,
    'subsample': 1,
    'sampling_method': 'uniform',
    'lambda': 2,
    'alpha': 1,
    'tree_method': 'auto'
}

num_boost_round = 100
watchlist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param,
                dtrain,
                num_boost_round,
                evals=watchlist,
                early_stopping_rounds=20)
# save model
bst.save_model('0002.model')

# load model
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('0002.model')

# pred
pred = bst.predict(dtest)

print(f'pred:{pred}')
print(f'true:{y_test}')
