#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:54:40 2019

@author: liweidong
"""

from __future__ import print_function

from sklearn import svm
from sklearn import preprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split


# 采样数量
samples_num = 40

# load data from csv
df=pd.DataFrame(pd.read_csv('data.csv',header=0))
print("加载数据成功")

file_columns = df.columns.tolist()
# header = 'type' 为类别列
data_y = df['type'].tolist()
# x reshape 
data_xx = np.zeros([df.shape[0],df.shape[1]//samples_num,samples_num])
for i in range(df.shape[0]):
    data_xx[i] = df.values[i][1:].reshape(df.shape[1]//40,40)

# 将矩阵一分为二, 两部分分别为触发前和触发后
data_xx1,data_xx2 = np.hsplit(data_xx,2)
# 将矩阵1翻转，成为触发前的时间序列，16通道，每个通道40个采样点,共5000组 (5000, 16, 40)
data_xx1_flip = np.flip(data_xx1,2)
print("数据构造完成")


## 数据清洗
# 利用各通道不同类别的的均值替换 缺失值空值
data_avg = np.zeros((3,data_xx1_flip.shape[1]))
data_sum = np.zeros((3,data_xx1_flip.shape[1]))
data_num = np.zeros((3,data_xx1_flip.shape[1]))

# 数据清洗准备：按通道统计总和及总量
for i in range(data_xx1_flip.shape[0]):
    for j in range(data_xx1_flip.shape[1]):
        for k in range(data_xx1_flip.shape[2]):
            # row i col j avg
            if data_y[i] == 0 and data_xx1_flip[i][j][k] != -999 and data_xx1_flip[i][j][k] != '' and not np.isnan(data_xx1_flip[i][j][k]):
                data_sum[0][j] += data_xx1_flip[i][j][k]
                data_num[0][j] += 1
            if data_y[i] == 1 and data_xx1_flip[i][j][k] != -999 and data_xx1_flip[i][j][k] != '' and not np.isnan(data_xx1_flip[i][j][k]):
                data_sum[1][j] += data_xx1_flip[i][j][k]
                data_num[1][j] += 1
            if data_y[i] == 2 and data_xx1_flip[i][j][k] != -999 and data_xx1_flip[i][j][k] != '' and not np.isnan(data_xx1_flip[i][j][k]):
                data_sum[2][j] += data_xx1_flip[i][j][k]
                data_num[2][j] += 1
# 数据清洗准备：均值 avg for every channel every class
for i in range(3):
    for j in range(data_xx1_flip.shape[1]):
        try:
            data_avg[i][j] = data_sum[i][j]/data_num[i][j]
        except ZeroDivisionError as e:
            print('统计数为零：',e)

# 数据清洗 avg replace null/nan/etc.
for i in range(data_xx1_flip.shape[0]):
    for j in range(data_xx1_flip.shape[1]):
        for k in range(data_xx1_flip.shape[2]):
            if data_y[i] == 0 and ( data_xx1_flip[i][j][k] == -999 or data_xx1_flip[i][j][k] == '' or np.isnan(data_xx1_flip[i][j][k]) ):
                data_xx1_flip[i][j][k] = data_avg[0][j]
            if data_y[i] == 1 and ( data_xx1_flip[i][j][k] == -999 or data_xx1_flip[i][j][k] == '' or np.isnan(data_xx1_flip[i][j][k]) ):
                data_xx1_flip[i][j][k] = data_avg[1][j]
            if data_y[i] == 2 and ( data_xx1_flip[i][j][k] == -999 or data_xx1_flip[i][j][k] == '' or np.isnan(data_xx1_flip[i][j][k]) ):
                data_xx1_flip[i][j][k] = data_avg[2][j]

print("数据清洗完成")

# 每个通道求均值作为特征
data_x = np.array([np.mean(data_xx1_flip[i],axis=1).tolist() for i in range(data_xx1_flip.shape[0])])
# 数据规范化，（0，1）规范化 
data_x=preprocessing.scale(data_x)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

C = 1.0  # SVM regularization parameter


model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)

print('开始训练...')
model = model.fit(x_train, y_train) 
score = model.score(x_test, y_test)

print('模型得分：',score)



