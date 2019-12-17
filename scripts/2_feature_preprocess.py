import os
import utils
import numpy as np
import pandas as pd
from imp import reload
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取数据集, 变量字典
raw_data = pd.read_csv('data/data.csv')
var_dict = pd.read_excel('result_files/feature_dict.xlsx')
var_dict.head()

# 生成统计描述文件feture_summary_all.xlsx
utils.eda(raw_data, var_dict)

# 找出取值有 {-999} 和 {nan} 所有的变量
missing_with_default = raw_data[raw_data == -999].count()
missing_with_default = list(missing_with_default[missing_with_default > 0].index.values)

missing_without_default = raw_data.isnull().sum()
missing_without_default = list(missing_without_default[missing_without_default > 0].index.values)

base_var = ['rxAudioKBitrate', 'memory_inactive', 'txVideoKBitrate', 'duration', 'cpuTotalUsage', 'rxVideoKBitrate', 'cpuAppUsage',
            'sentBitrate', 'txAudioKBitrate', 'memory_free', 'memory_app_used', 'sentFrameRate', 'lag', 'fps', 'cpu', 'userCount']

# plot missing counts distribution of -999 and nan
"""
经观察每个特征的80次观察中, X_lead_1 和 X_lag_1均未缺失
且nan的observation仅占总数据集的0.5%.
考虑从训练集中drop掉nan的observation.
"""
utils.plot_missing_dist(base_var, raw_data, False)
utils.plot_missing_dist(base_var, raw_data)

# 观察删除nan后数据集大小
X_without_missing = raw_data.dropna(how='any')
X_without_missing.shape # (4972, 1281)
print((raw_data.shape[0]-X_without_missing.shape[0]) * 100/raw_data.shape[0]) # 0.56%
utils.eda(X_without_missing, var_dict, save_label='drop_na')

# 数据类型转换
X_without_missing = X_without_missing.astype('float64')
X_without_missing['type'] = X_without_missing['type'].astype('int16')
X_without_missing.dtypes

# 划分训练集与测试集, dropna
X_wom_train, X_wom_test, y_wom_train, y_wom_test = train_test_split(X_without_missing[var_dict['var_code']]\
                                                                    , X_without_missing['type']\
                                                                    , test_size = 0.2\
                                                                    , random_state = 43)
X_wom_train.shape, X_wom_test.shape, y_wom_train.shape, y_wom_test.shape
pd.crosstab(y_wom_train, 1)/y_wom_train.shape[0] #(0,0.790546),(1,0.074931),(2,0.134524)
pd.crosstab(y_wom_test, 1)/y_wom_test.shape[0] #(0,0.789950),(1,0.085427),(2,0.124623)

# 变量衍生
# duration：通话时长，单位为秒，累计值；重置链接后清零。
# txAudioKBitrate:音频发送码率 (Kbps)，瞬时值
# rxAudioKBitrate:音频接收码率 (Kbps)，瞬时值
# txVideoKBitrate:音频发送码率 (Kbps)，瞬时值
# rxVideoKBitrate:音频接收码率 (Kbps)，瞬时值
# cpuTotalUsage:当前系统的 CPU 使用率 (%)
# cpuAppUsage:当前 App 的 CPU 使用率 (%)
# userCount: 当前频道内的用户人数
# sentFrameRate: 不重要
# sentBitrate: 不重要
# cpu: 上报数列的最高值
# lag: 客户端与game server的ping值
# fps: 客户端的针率
# memory_free：客户端未使用
# memory_app_used
# memory_inactive:
X_wom_train = utils.generate_feature(X_wom_train, var_dict, base_var)
X_wom_test = utils.generate_feature(X_wom_test, var_dict, base_var)
X_wom_train.loc[y_wom_train.index.values,'type'] = y_wom_train
X_wom_test.loc[y_wom_test.index.values, 'type'] = y_wom_test
utils.save_data_to_pickle(X_wom_train,'data','X_wom_train_with_y.pkl')
utils.save_data_to_pickle(X_wom_test,'data','X_wom_test_with_y.pkl')

# 划分训练集与测试集, 缺失处理成-999
raw_data = raw_data.fillna(-999.0)
raw_data = raw_data.astype('float64')
raw_data['type'] = raw_data['type'].astype('int16')
X_train, X_test, y_train, y_test = train_test_split(
    raw_data[var_dict['var_code']], raw_data['type'], test_size=0.2, random_state=43)
# ((4000, 1280), (4000,), (1000, 1280), (1000,))
X_train.shape, y_train.shape, X_test.shape, y_test.shape
pd.crosstab(y_train, 1)/y_train.shape[0]  # (0,0.78575),(1,0.07725),(2,0.13700)
pd.crosstab(y_test, 1)/y_test.shape[0]  # (0,0.801),(1,0.079),(2,0.120)

X_train = utils.generate_feature(X_train, var_dict, base_var)
X_test = utils.generate_feature(X_test, var_dict, base_var)
X_train.shape, X_test.shape
X_train.loc[y_train.index.values,'type'] = y_train
X_test.loc[y_test.index.values, 'type'] = y_test
utils.save_data_to_pickle(X_train,'data','X_train_with_y.pkl')
utils.save_data_to_pickle(X_test,'data','X_test_with_y.pkl')