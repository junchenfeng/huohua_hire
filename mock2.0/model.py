# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

#新数据
data_notna = pd.read_csv("C:/Users/Administrator/Desktop/Sunday/data_notna.csv")
#拆分变量，响应变量需要把label=2换成1
X = data_notna.drop(['type'],axis=1)
y = data_notna['type'].replace(2,1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#数据标准化
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)
#建模
lr = LR()
roc_auc = cross_val_score(lr,X_train_std,y_train,cv=5,scoring='roc_auc')
print('roc_auc:',roc_auc.mean(),roc_auc)
#roc_auc: 0.8296151033710977 [0.81343979 0.83254629 0.82271343 0.8609671  0.8184089 ]
#skfold = StratifiedKFold(n_splits = 5,random_state = 0)

#先调两个参数
penaltys = ['l1','l2']
#Cs = [0.001,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10],输出0.1和l1,0.8875566455991029
Cs = np.linspace(0.01,0.2,20)  
param = dict(penalty = penaltys, C = Cs)
gsCV = GridSearchCV(lr,param,cv=5,scoring='roc_auc')
gsCV.fit(X_train_std,y_train)
print(gsCV.best_score_, gsCV.best_params_, gsCV.grid_scores_)
#结果：0.08和l1,0.8884890903752685

#test_means = gsCV.cv_results_['mean_test_score']
#test_stds = gsCV.cv_results_['std_test_score']
#train_means = gsCV.cv_results_['mean_train_score']
#train_stds = gsCV.cv_results_['std_train_score']

#代入最优参数，考虑类不平衡问题，微调后得分变高了一点点
lr_l1 = LR(penalty='l1',C = 0.08)
lr_l1.fit(X_train_std,y_train)
lr_l1.score(X_train_std,y_train) #0.9329825533093326
lr_b = LR(penalty='l1',C = 0.08,class_weight = 'balanced')
lr_b.fit(X_train_std,y_train)
lr_b.score(X_train_std,y_train) #0.9160897258377181
lr_w = LR(penalty='l1',C = 0.08,class_weight = {0:0.4,1:0.6})
lr_w.fit(X_train_std,y_train)
lr_w.score(X_train_std,y_train) #0.9335364165051232

#最终选用参数l1惩罚，C=0.08，类权重为2:3，模型评估得分
precision = cross_val_score(lr_w,X_train_std,y_train,cv=5,scoring='precision')
recall = cross_val_score(lr_w,X_train_std,y_train,cv=5,scoring='recall')
f1 = cross_val_score(lr_w,X_train_std,y_train,cv=5,scoring='f1')
roc_auc = cross_val_score(lr_w,X_train_std,y_train,cv=5,scoring='roc_auc')
print('precision:',precision_w.mean(),precision_w)
print('recall:',recall.mean(),recall)
print('f1:',f1.mean(),f1)
print('roc_auc:',roc_auc.mean(),roc_auc)
#precision: 0.8259307056944115 [0.75268817 0.86075949 0.825      0.86842105 0.82278481]
#recall: 0.6466766243465274 [0.67307692 0.65384615 0.63461538 0.6407767  0.63106796]
#f1: 0.7245872967231135 [0.7106599  0.7431694  0.7173913  0.73743017 0.71428571]
#roc_auc: 0.8887237676204691 [0.89137256 0.89256866 0.87795141 0.89746128 0.88426493]
#调参后的roc_auc得分比原始提高约0.06

#直接查看模型评估精确率与准确率
pred_train = lr_w.predict(X_train_std)
print(precision_score(y_train, pred_train),recall_score(y_train, pred_train))
#精确率0.8341346153846154 召回率0.6698841698841699
#模型应用到测试集
pred_test = lr_w.predict(X_test_std)
report_test = classification_report(y_test, pred_test)
cm_test = confusion_matrix(y_test, pred_test)
print('测试集分类报告：\n',report_test)
print('测试集混淆矩阵：\n',cm_test)

#计算AUC
pred_0=list(pred[:,1])
fpr,tpr,thresholds=roc_curve(y_test,pred_0)  
auc=roc_auc_score(y_test,pred_0) #0.8883770161290322
#ROC曲线图
plt.figure()
plt.plot(fpr,tpr)
plt.title('$ROC curve$')
plt.show()    
#最佳阈值
KS_max=0
best_thr=0
for i in range(len(fpr)):
    if(i==0):
        KS_max=tpr[i]-fpr[i]
        best_thr=thresholds[i]
    elif (tpr[i]-fpr[i]>KS_max):
        KS_max = tpr[i] - fpr[i]
        best_thr = thresholds[i]
print('最大KS为：',KS_max)
print('最佳阈值为：',best_thr)
#最大KS为： 0.7453326612903226
#最佳阈值为： 0.21169786286332753

#输出测试集分类概率
pred = lr_w.predict_proba(X_test_std)
result = pd.DataFrame(pred)

