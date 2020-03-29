# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#数据
data = pd.read_csv("C:/Users/Administrator/Desktop/Sunday/data.csv")
#删除不重要变量
not_important = data.columns.str.contains('_lag_')+data.columns.str.contains('duration_')+data.columns.str.contains('sentFrameRate_')+data.columns.str.contains('sentBitrate_')+data.columns.str.contains('memory_app_used_')+data.columns.str.contains('memory_inactive_')
data_sample = data.loc[:,~not_important]

#修改步骤：删除轻微抖动课堂样本
data_del = data_sample[data_sample['type'] != 1]

#剩余自变量间的相关性，以最接近工单提交的数据X_lead_1为例
lead_1 = data_del.columns.str.endswith('_lead_1')
data_lead_1 = data_del.loc[:,lead_1]
data_lead_1.corr()
#发现cpuTotalUsage_lead_1与cpuAppUsage_lead_1（0.59）以及cpu_lead_1（0.72）相关系数较大，去掉cpuTotalUsage
correlation = data_del.columns.str.contains('cpuTotalUsage_')
data_remain = data_del.loc[:,~correlation]
data_remain.head()

#删除缺失值
data_remain[data_remain == -999] = np.nan
data_notna = data_remain.dropna()
data_notna.groupby('type').size()

data_notna.to_csv('data_notna.csv')


#原始样本量：正常课堂0:3944；抖动课堂1：388；卡顿课堂2:668
#剩余样本量：正常课堂0:3868；卡顿课堂2:646

















