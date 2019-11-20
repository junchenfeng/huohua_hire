'''
author: Li Weidong

copy from keras/mnist_cnn.py
used to solve the huohua_hire problem

model train 
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 3
epochs = 20
# 采样数量
samples_num = 40

# input image dimensions
img_rows, img_cols = 16, 40

# load data from csv
df=pd.DataFrame(pd.read_csv('data.csv',header=0))
print("加载数据成功")

file_columns = df.columns.tolist()
# header = 'type' 为类别列
data_y = df['type'].tolist()
# x reshape 
# data_x = np.zeros([df.shape[0],df.shape[1]//samples_num,samples_num])
data_xx = np.zeros([df.shape[0],df.shape[1]//samples_num,samples_num])
for i in range(df.shape[0]):
    data_xx[i] = df.values[i][1:].reshape(df.shape[1]//40,40)

# 将矩阵一分为二，以便进行x重组。利用了数据通道的规律
data_xx1,data_xx2 = np.hsplit(data_xx,2)
# 将矩阵1翻转，以便和矩阵2拼接后形成时间序列
data_xx1_flip = np.flip(data_xx1,2)
# 形成时间序列。触发时机不具有稳定性，所以将一个通道的整体采样作为一个样本
#data_x=np.concatenate((data_xx1_flip,data_xx2),axis=2)
data_x = data_xx1_flip
print("数据构造完成")


## 数据清洗
# 利用各通道不同类别的的均值替换 缺失值空值
data_avg = np.zeros((3,data_x.shape[1]))
data_sum = np.zeros((3,data_x.shape[1]))
data_num = np.zeros((3,data_x.shape[1]))

# 数据清洗准备：按通道统计总和及总量
for i in range(data_x.shape[0]):
    for j in range(data_x.shape[1]):
        for k in range(data_x.shape[2]):
            # row i col j avg
            if data_y[i] == 0 and data_x[i][j][k] != -999 and data_x[i][j][k] != '' and not np.isnan(data_x[i][j][k]):
                data_sum[0][j] += data_x[i][j][k]
                data_num[0][j] += 1
            if data_y[i] == 1 and data_x[i][j][k] != -999 and data_x[i][j][k] != '' and not np.isnan(data_x[i][j][k]):
                data_sum[1][j] += data_x[i][j][k]
                data_num[1][j] += 1
            if data_y[i] == 2 and data_x[i][j][k] != -999 and data_x[i][j][k] != '' and not np.isnan(data_x[i][j][k]):
                data_sum[2][j] += data_x[i][j][k]
                data_num[2][j] += 1
# 数据清洗准备：均值 avg for every channel every class
for i in range(3):
    for j in range(data_x.shape[1]):
        try:
            data_avg[i][j] = data_sum[i][j]/data_num[i][j]
        except ZeroDivisionError as e:
            print('统计数为零：',e)

# 数据清洗 avg replace null/nan/etc.
for i in range(data_x.shape[0]):
    for j in range(data_x.shape[1]):
        for k in range(data_x.shape[2]):
            if data_y[i] == 0 and ( data_x[i][j][k] == -999 or data_x[i][j][k] == '' or np.isnan(data_x[i][j][k]) ):
                data_x[i][j][k] = data_avg[0][j]
            if data_y[i] == 1 and ( data_x[i][j][k] == -999 or data_x[i][j][k] == '' or np.isnan(data_x[i][j][k]) ):
                data_x[i][j][k] = data_avg[1][j]
            if data_y[i] == 2 and ( data_x[i][j][k] == -999 or data_x[i][j][k] == '' or np.isnan(data_x[i][j][k]) ):
                data_x[i][j][k] = data_avg[2][j]

print("数据清洗完成")
## 将任务按照图像分类来做
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("训练测试数据准备完成")

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', # Sigmoid
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print("模型搭建完成，开始训练")
print("--"*20)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('hhsw.h5')

print("模型训练完成，开始评估")
score = model.evaluate(x_test, y_test, verbose=0)
#score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
