# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('scripts')
import utils
import copy
import numpy as np
import pandas as pd
from imp import reload
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from functools import partial
from sklearn.svm import SVC
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample as hp_sample
from sklearn.preprocessing import normalize, scale, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier #0.22新增加的

"""
Step1. load data and feature dict
"""
# load new feature summary, 每一轮选出63个变量
feature_dict = pd.read_excel('result_files/feature_importances_v2.xlsx')
feature_dict.set_index('var_code', inplace=True)
selected_features = list(feature_dict[feature_dict['exclude'].isnull()].index.values)
selected_features
len(selected_features) #63

# load data
x_train_v2 = utils.load_data_from_pickle('data', 'x_train_770.pkl')
x_test_v2 = utils.load_data_from_pickle('data', 'x_test_770.pkl')
y_train = utils.load_data_from_pickle('data', 'y_train.pkl')
y_test = utils.load_data_from_pickle('data', 'y_test.pkl')

"""
Step2. tune parameters, fit model
"""
reload(utils)
utils.draw_hp_sample()
best, trials = utils.hyper_best(x_train_v2[selected_features], y_train, max_evals=3000)
benchmark_best = best
benchmark_trials = trials

benchmark_best_param = {
        'booster': 'gblinear'
        , 'colsample_bylevel': 0.157
        , 'colsample_bynode': 0.224
        , 'colsample_bytree': 0.246
        , 'gamma': 0.1
        , 'learning_rate': 0.191
        , 'n_estimators': 16
        , 'max_depth': 4
        , 'min_child_weight': 1
        , 'subsample': 0.625
        , 'objective': 'multi:softmax'
        , 'eval_metric': 'mlogloss'
        , 'num_class': 3
        }

reload(utils)
# 绘制参数与损失函数迭代图
utils.plot_best_param(3, 5, benchmark_best, benchmark_trials)

xgb_clf = xgb.XGBClassifier(**benchmark_best_param)
xgb_clf.fit(x_train_v2[selected_features], y_train, \
                  eval_set=[(x_train_v2[selected_features], y_train), (x_test_v2[selected_features], y_test)],\
                  eval_metric='mlogloss', verbose=True)
y_pred = xgb_clf.predict(x_test_v2[selected_features])

# 绘制损失函数图
utils.plot_logloss(xgb_clf)

# 准备预测准确率
all_y = np.vstack([y_test, y_pred])
all_y_df = pd.Series(y_test, index=y_test.index, name='y_test')
all_y_df = all_y_df.to_frame()
all_y_df['y_pred'] = pd.Series(y_pred, index=y_test.index)

result_df = (pd.crosstab([all_y_df['y_test'],all_y_df['y_pred']], 'count')).reset_index()
accuracy = result_df[result_df.y_test == result_df.y_pred]
accuracy

# 画准确率图
score = accuracy_score(y_test, y_pred)
ax = sns.countplot(y_test,label="ground_true_y")
sns.lineplot(x="y_test", y="count", data=accuracy, color="coral", label='correct_y_pred')
plt.title('accuracy=${}$, total_sample=${}$'.format(score, len(y_test)))
plt.savefig('images/accuracy_score.png')

# 保存模型
xgb_clf.save_model('data/xgb_model.pkl')