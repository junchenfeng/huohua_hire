import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('scripts')
import utils
import numpy as np
import pandas as pd
from imp import reload
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
#reload(utils)

all_wom_data = utils.load_data_from_pickle('data','all_wom_data.pkl')
var_dict = pd.read_excel('result_files/feature_dict.xlsx')

"""
特征衍生
"""
# 划分训练集与测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_wom_data[var_dict['var_code']],all_wom_data['type'],test_size=0.2,random_state=utils.SEED)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# 分别对训练集与测试集进行特征衍生 prepare base 14 features
reload(utils)
import copy
base_var = copy.deepcopy(utils.BASE_VAR)
base_var.remove('sentFrameRate')
base_var.remove('sentBitrate')
base_var

# respectively generate feature by train and test
x_train_v1 = utils.generate_feature(x_train, var_dict, base_var) #缺失值-999也在转换的范围内
x_test_v1 = utils.generate_feature(x_test, var_dict, base_var)
x_train_v1.shape

# double check feature generation is fine
x_train_v1.isnull().sum().sum()
x_test_v1.isnull().sum().sum()

# we've got 770 new features here
additive_vars = list(set(x_train_v1.columns.values) - set(var_dict['var_code'])) 
len(additive_vars) #770

# train/test with new 770 features
x_train_v2 = x_train_v1[additive_vars]
x_test_v2 = x_test_v1[additive_vars]
x_train_v2.shape, x_test_v2.shape
x_train_v2.columns.values 
(x_train_v2 == np.inf).sum().sum()

# 存储数据集
utils.save_data_to_pickle(x_train_v2, 'data', 'x_train_770.pkl')
utils.save_data_to_pickle(x_test_v2, 'data', 'x_test_770.pkl')
utils.save_data_to_pickle(x_train_v1, 'data', 'x_train_v1.pkl')
utils.save_data_to_pickle(x_test_v1, 'data', 'x_test_v1.pkl')
utils.save_data_to_pickle(y_train, 'data', 'y_train.pkl')
utils.save_data_to_pickle(y_test, 'data', 'y_test.pkl')

# 通过模型计算变量重要性
from sklearn.ensemble import RandomForestClassifier #0.22新增加的
from sklearn.inspection import permutation_importance

"""
特征选择(770个变量中选择)
"""
# 特征summary
reload(utils)
summary = utils.feature_range(x_train_v2)
summary.sort_values('entropy',ascending=True,inplace=True)
summary.set_index('var_code',inplace=True)
summary.head()
summary.sort_values('missing',ascending=False).head(50)
summary.sort_values('uniques').head(50)

# 提取16个特征的衍生变量组
feature_group= []
for var in base_var:
    feature_group.append([var_code for var_code in summary.index.values if var in var_code])
    
# cpu出现的特征单独处理一次
len(feature_group)
del_f = [f for f in feature_group[10] if 'cpuTotal' in f or 'cpuApp' in f]
del_f
len(feature_group[10])
len(del_f)
len(feature_group[1])
feature_group[10] = list(set(feature_group[10]) - set(del_f))

# lag出现的特征单独处理一次
for i in range(len(feature_group)):
    print(len(feature_group[i]),feature_group[i][0])
lag_group = [f for f in feature_group[11] if 'lag' == f.split('_')[0]]
len(lag_group)
feature_group[11] = lag_group

summary.loc[feature_group[0],'entropy']
feature_group[7]

# 每个特征组使用层次聚类查看一下变量聚类及变量相关性
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
for ith in range(len(feature_group)):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(50, 55))
    corr = spearmanr(x_train_v2[feature_group[ith]]).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=x_train_v2[feature_group[ith]].columns.values, ax=ax1, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    
    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    plt.suptitle('{}_dendrogram'.format(feature_group[ith][0]))
    fig.tight_layout()
    plt.savefig('images/{}_feature_correlation.png'.format(feature_group[ith][0]))
    
# 分别计算16组特征的重要性: ranomforest_importance 和 permutation_importance
for vth in range(len(feature_group)):
    clf_v1 = RandomForestClassifier(n_estimators=150, random_state=utils.SEED)
    clf_v1 = clf_v1.fit(x_train_v2[feature_group[vth]],y_train)
    
    per_imp = permutation_importance(clf_v1, x_train_v2[feature_group[vth]], y_train, n_repeats=10, random_state=utils.SEED)
    summary.loc[feature_group[vth],'group_rf_imp'] = pd.Series(clf_v1.feature_importances_, index=x_train_v2[feature_group[vth]].columns.values)
    summary.loc[feature_group[vth],'group_per_imp'] = pd.Series(per_imp.importances_mean.ravel(), index=x_train_v2[feature_group[vth]].columns.values)
    
    perm_sorted_idx = per_imp.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(clf_v1.feature_importances_)
    tree_indices = np.arange(0, len(clf_v1.feature_importances_)) + 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,clf_v1.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(x_train_v2[feature_group[vth]].columns.values)
    ax1.set_yticks(tree_indices)
    ax1.set_ylim((0, len(clf_v1.feature_importances_)))
    ax1.set_title('rondom_forest_importance')
    ax2.boxplot(per_imp.importances[perm_sorted_idx].T, vert=False, labels=x_train_v2[feature_group[vth]].columns.values)
    ax2.set_title('permutation_importance')
    plt.suptitle('RandomForest Importance VS Permutation Importances')
    plt.tight_layout()
    plt.savefig('images/{}_importance_comparison_between_rf_and_permutation.png'.format(feature_group[vth][0]))

# 计算方差
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.1)
sel.fit_transform(x_train_v2)
summary.set_index('var_code',inplace=True)
summary['variances'] = pd.Series(sel.variances_, index=x_train_v2.columns.values)
summary.head()

# 所有特征的randomforest_importance 和 permutation_importance
clf_v2 = RandomForestClassifier(n_estimators=200, random_state=utils.SEED)
clf_v2 = clf_v2.fit(x_train_v2,y_train)

per_imp = permutation_importance(clf_v2, x_train_v2, y_train, n_repeats=15, random_state=utils.SEED)
per_imp.importances_mean

summary['rf_imp'] = pd.Series(clf_v2.feature_importances_, index=x_train_v2.columns.values)
summary['per_imp'] = pd.Series(per_imp.importances_mean, index=x_train_v2.columns.values)
summary['dtypes'] = summary['dtypes'].astype('str')
summary.loc[feature_dict['group_rf_imp'] < 0.03,'exclude'] = 'group_rf_imp<0.03'
summary.to_excel('result_files/feature_importances_v2.xlsx')