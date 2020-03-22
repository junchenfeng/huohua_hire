
产品方案

一、引言
在上课过程中，如果老师发现学生卡住了（或者学生告诉老师自己不能看到老师或无法操作课件），就会向技术支持提交一个网络状况工单，根据监课情况，工单原因包括：轻微抖动不影响上课和网络卡顿。为降低网络工单率，需要在工单提交时可以预测网络工单的关闭原因，这样就可以在老师提交工单时在后台直接关闭预测为“轻微抖动不影响上课”的工单，而仅放行“网络卡顿”的工单。

二、产品简介
当教师点击提交网络状况工单后即触发该模型，通过提交工单前一段时间内所记录的音频、视频及师生间课件控制的数据，可以得到该堂课属于正常课堂还是问题课堂的概率，由此判断该堂课是否为网络卡顿，进而触发工单的放行和关闭状态。

三、实现原理
通过逻辑回归方法，利用课堂的网络监控数据，对可能有网络问题的课堂进行分类，判断其是否具有网络卡顿问题。该模型优点为分类结果一目了然且能得到属于某一类的概率，可解释性强，运行速度快。

四、业务流程
需要在教师提交网络状况工单时触发模型计算，返回结果触发工单自动关闭操作。

五、模型评估
当前模型把课堂分为三种：正常课堂、轻微抖动不影响和网络卡顿，实际上前两种结果均可使得工单关闭。整体上看，模型的精确率和召回率在83%-84%，若把前两种结果合并，则模型的精确率和召回率可达到94%，即预测为非网络卡顿的课堂中，有94%的课堂确实没有网络卡顿，同时在真的没有网络卡顿问题的课堂中我们能预测正确的比例为94%。故我们的预测效果是不错的，能在一定程度上提高工单效率。

六、结论
该模型能在很大程度上提高网络工单处理率，进而降低网络工单率，实用性强，建议排期进开发。


```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
```


```python
#原始数据
data = pd.read_csv("C:/Users/Administrator/Desktop/Sunday/data.csv")
```


```python
#数据概况
data.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>txAudioKBitrate_lead_1</th>
      <th>txAudioKBitrate_lead_2</th>
      <th>txAudioKBitrate_lead_3</th>
      <th>txAudioKBitrate_lead_4</th>
      <th>txAudioKBitrate_lead_5</th>
      <th>txAudioKBitrate_lead_6</th>
      <th>txAudioKBitrate_lead_7</th>
      <th>txAudioKBitrate_lead_8</th>
      <th>txAudioKBitrate_lead_9</th>
      <th>...</th>
      <th>memory_inactive_lag_31</th>
      <th>memory_inactive_lag_32</th>
      <th>memory_inactive_lag_33</th>
      <th>memory_inactive_lag_34</th>
      <th>memory_inactive_lag_35</th>
      <th>memory_inactive_lag_36</th>
      <th>memory_inactive_lag_37</th>
      <th>memory_inactive_lag_38</th>
      <th>memory_inactive_lag_39</th>
      <th>memory_inactive_lag_40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>54</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>554680320.0</td>
      <td>556302336.0</td>
      <td>554041344.0</td>
      <td>556220416.0</td>
      <td>560152576.0</td>
      <td>561233920.0</td>
      <td>560857088.0</td>
      <td>554582016.0</td>
      <td>554614784.0</td>
      <td>554598400.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>54</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>523796480.0</td>
      <td>525516800.0</td>
      <td>527368192.0</td>
      <td>527826944.0</td>
      <td>527974400.0</td>
      <td>527728640.0</td>
      <td>529203200.0</td>
      <td>531628032.0</td>
      <td>530579456.0</td>
      <td>527695872.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>655065088.0</td>
      <td>654704640.0</td>
      <td>653983744.0</td>
      <td>653934592.0</td>
      <td>653082624.0</td>
      <td>656326656.0</td>
      <td>654409728.0</td>
      <td>655081472.0</td>
      <td>655851520.0</td>
      <td>654884864.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>550387712.0</td>
      <td>-999.0</td>
      <td>551092224.0</td>
      <td>-999.0</td>
      <td>551010304.0</td>
      <td>-999.0</td>
      <td>549273600.0</td>
      <td>-999.0</td>
      <td>549421056.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>53</td>
      <td>12.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>51.0</td>
      <td>43.0</td>
      <td>42.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>585465856.0</td>
      <td>583483392.0</td>
      <td>582746112.0</td>
      <td>585875456.0</td>
      <td>585482240.0</td>
      <td>585564160.0</td>
      <td>586432512.0</td>
      <td>586563584.0</td>
      <td>586448896.0</td>
      <td>586891264.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>451870720.0</td>
      <td>-999.0</td>
      <td>451788800.0</td>
      <td>-999.0</td>
      <td>451788800.0</td>
      <td>-999.0</td>
      <td>451788800.0</td>
      <td>-999.0</td>
      <td>451788800.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>53</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>372244480.0</td>
      <td>-999.0</td>
      <td>372244480.0</td>
      <td>-999.0</td>
      <td>372244480.0</td>
      <td>-999.0</td>
      <td>372244480.0</td>
      <td>-999.0</td>
      <td>372244480.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>7</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>54</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>513966080.0</td>
      <td>-999.0</td>
      <td>514277376.0</td>
      <td>-999.0</td>
      <td>514523136.0</td>
      <td>-999.0</td>
      <td>514080768.0</td>
      <td>-999.0</td>
      <td>514621440.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>671203328.0</td>
      <td>-999.0</td>
      <td>667041792.0</td>
      <td>-999.0</td>
      <td>663519232.0</td>
      <td>-999.0</td>
      <td>663764992.0</td>
      <td>-999.0</td>
      <td>659046400.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>48.0</td>
      <td>49.0</td>
      <td>51.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>689881088.0</td>
      <td>-999.0</td>
      <td>689848320.0</td>
      <td>-999.0</td>
      <td>690012160.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>689668096.0</td>
      <td>689635328.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>589414400.0</td>
      <td>-999.0</td>
      <td>587841536.0</td>
      <td>587890688.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>588398592.0</td>
      <td>588316672.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>612515840.0</td>
      <td>-999.0</td>
      <td>612171776.0</td>
      <td>-999.0</td>
      <td>609271808.0</td>
      <td>-999.0</td>
      <td>612007936.0</td>
      <td>-999.0</td>
      <td>612384768.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>52</td>
      <td>46.0</td>
      <td>47.0</td>
      <td>40.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>42.0</td>
      <td>50.0</td>
      <td>...</td>
      <td>591773696.0</td>
      <td>590282752.0</td>
      <td>589709312.0</td>
      <td>589250560.0</td>
      <td>589119488.0</td>
      <td>588726272.0</td>
      <td>589201408.0</td>
      <td>586186752.0</td>
      <td>590479360.0</td>
      <td>590610432.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>44</td>
      <td>43.0</td>
      <td>48.0</td>
      <td>44.0</td>
      <td>52.0</td>
      <td>49.0</td>
      <td>51.0</td>
      <td>52.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>559972352.0</td>
      <td>560513024.0</td>
      <td>558481408.0</td>
      <td>558776320.0</td>
      <td>560431104.0</td>
      <td>559939584.0</td>
      <td>555827200.0</td>
      <td>556220416.0</td>
      <td>559939584.0</td>
      <td>558448640.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>53</td>
      <td>52.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>372129792.0</td>
      <td>-999.0</td>
      <td>372097024.0</td>
      <td>-999.0</td>
      <td>371900416.0</td>
      <td>-999.0</td>
      <td>373260288.0</td>
      <td>-999.0</td>
      <td>373014528.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>388186112.0</td>
      <td>-999.0</td>
      <td>388169728.0</td>
      <td>-999.0</td>
      <td>386744320.0</td>
      <td>-999.0</td>
      <td>419987456.0</td>
      <td>-999.0</td>
      <td>419725312.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>566132736.0</td>
      <td>-999.0</td>
      <td>565985280.0</td>
      <td>-999.0</td>
      <td>563904512.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>567066624.0</td>
      <td>-999.0</td>
      <td>568344576.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>631160832.0</td>
      <td>-999.0</td>
      <td>631570432.0</td>
      <td>-999.0</td>
      <td>629473280.0</td>
      <td>-999.0</td>
      <td>629735424.0</td>
      <td>-999.0</td>
      <td>629735424.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>0.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>513556480.0</td>
      <td>513490944.0</td>
      <td>513277952.0</td>
      <td>515997696.0</td>
      <td>515997696.0</td>
      <td>513343488.0</td>
      <td>517324800.0</td>
      <td>514277376.0</td>
      <td>513441792.0</td>
      <td>514310144.0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 1281 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Columns: 1281 entries, type to memory_inactive_lag_40
    dtypes: float64(1221), int64(60)
    memory usage: 48.9 MB
    


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>txAudioKBitrate_lead_1</th>
      <th>txAudioKBitrate_lead_2</th>
      <th>txAudioKBitrate_lead_3</th>
      <th>txAudioKBitrate_lead_4</th>
      <th>txAudioKBitrate_lead_5</th>
      <th>txAudioKBitrate_lead_6</th>
      <th>txAudioKBitrate_lead_7</th>
      <th>txAudioKBitrate_lead_8</th>
      <th>txAudioKBitrate_lead_9</th>
      <th>...</th>
      <th>memory_inactive_lag_31</th>
      <th>memory_inactive_lag_32</th>
      <th>memory_inactive_lag_33</th>
      <th>memory_inactive_lag_34</th>
      <th>memory_inactive_lag_35</th>
      <th>memory_inactive_lag_36</th>
      <th>memory_inactive_lag_37</th>
      <th>memory_inactive_lag_38</th>
      <th>memory_inactive_lag_39</th>
      <th>memory_inactive_lag_40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.00000</td>
      <td>5000.000000</td>
      <td>4998.000000</td>
      <td>4998.000000</td>
      <td>4998.000000</td>
      <td>4997.000000</td>
      <td>4997.000000</td>
      <td>4997.000000</td>
      <td>4994.000000</td>
      <td>4992.000000</td>
      <td>...</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
      <td>4.998000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.34480</td>
      <td>48.327200</td>
      <td>48.554222</td>
      <td>48.675270</td>
      <td>48.987595</td>
      <td>49.178507</td>
      <td>49.400240</td>
      <td>49.423054</td>
      <td>49.303564</td>
      <td>49.288061</td>
      <td>...</td>
      <td>5.016824e+08</td>
      <td>5.017613e+08</td>
      <td>5.016951e+08</td>
      <td>5.018133e+08</td>
      <td>5.016922e+08</td>
      <td>5.019673e+08</td>
      <td>5.012356e+08</td>
      <td>5.020848e+08</td>
      <td>5.011551e+08</td>
      <td>5.020959e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.70229</td>
      <td>37.568168</td>
      <td>37.449609</td>
      <td>37.332675</td>
      <td>37.184188</td>
      <td>37.086354</td>
      <td>36.977244</td>
      <td>36.966075</td>
      <td>39.819943</td>
      <td>39.810616</td>
      <td>...</td>
      <td>1.080941e+08</td>
      <td>1.078881e+08</td>
      <td>1.082330e+08</td>
      <td>1.082250e+08</td>
      <td>1.082896e+08</td>
      <td>1.077850e+08</td>
      <td>1.092522e+08</td>
      <td>1.074560e+08</td>
      <td>1.091399e+08</td>
      <td>1.070901e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>...</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
      <td>-9.990000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.00000</td>
      <td>50.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>51.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>4.322714e+08</td>
      <td>4.316815e+08</td>
      <td>4.318290e+08</td>
      <td>4.319601e+08</td>
      <td>4.318863e+08</td>
      <td>4.319396e+08</td>
      <td>4.326482e+08</td>
      <td>4.322181e+08</td>
      <td>4.315013e+08</td>
      <td>4.327014e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.00000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>...</td>
      <td>5.047501e+08</td>
      <td>5.045453e+08</td>
      <td>5.049958e+08</td>
      <td>5.051187e+08</td>
      <td>5.041684e+08</td>
      <td>5.047665e+08</td>
      <td>5.048238e+08</td>
      <td>5.047910e+08</td>
      <td>5.044060e+08</td>
      <td>5.046518e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.00000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>...</td>
      <td>5.795676e+08</td>
      <td>5.794734e+08</td>
      <td>5.795103e+08</td>
      <td>5.796086e+08</td>
      <td>5.796618e+08</td>
      <td>5.794161e+08</td>
      <td>5.796086e+08</td>
      <td>5.797233e+08</td>
      <td>5.798257e+08</td>
      <td>5.797069e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.00000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>71.000000</td>
      <td>70.000000</td>
      <td>...</td>
      <td>9.460122e+08</td>
      <td>9.459794e+08</td>
      <td>9.465037e+08</td>
      <td>9.464873e+08</td>
      <td>9.246310e+08</td>
      <td>9.247621e+08</td>
      <td>9.359852e+08</td>
      <td>9.360015e+08</td>
      <td>9.353789e+08</td>
      <td>9.358377e+08</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 1281 columns</p>
</div>




```python
data.groupby('type').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>txAudioKBitrate_lead_1</th>
      <th>txAudioKBitrate_lead_2</th>
      <th>txAudioKBitrate_lead_3</th>
      <th>txAudioKBitrate_lead_4</th>
      <th>txAudioKBitrate_lead_5</th>
      <th>txAudioKBitrate_lead_6</th>
      <th>txAudioKBitrate_lead_7</th>
      <th>txAudioKBitrate_lead_8</th>
      <th>txAudioKBitrate_lead_9</th>
      <th>txAudioKBitrate_lead_10</th>
      <th>...</th>
      <th>memory_inactive_lag_31</th>
      <th>memory_inactive_lag_32</th>
      <th>memory_inactive_lag_33</th>
      <th>memory_inactive_lag_34</th>
      <th>memory_inactive_lag_35</th>
      <th>memory_inactive_lag_36</th>
      <th>memory_inactive_lag_37</th>
      <th>memory_inactive_lag_38</th>
      <th>memory_inactive_lag_39</th>
      <th>memory_inactive_lag_40</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3944</td>
      <td>3943</td>
      <td>3943</td>
      <td>3943</td>
      <td>3943</td>
      <td>3943</td>
      <td>3943</td>
      <td>3940</td>
      <td>3940</td>
      <td>3940</td>
      <td>...</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
      <td>3944</td>
    </tr>
    <tr>
      <th>1</th>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>388</td>
      <td>387</td>
      <td>387</td>
      <td>...</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
      <td>386</td>
    </tr>
    <tr>
      <th>2</th>
      <td>668</td>
      <td>667</td>
      <td>667</td>
      <td>667</td>
      <td>666</td>
      <td>666</td>
      <td>666</td>
      <td>666</td>
      <td>665</td>
      <td>664</td>
      <td>...</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
      <td>668</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 1280 columns</p>
</div>




```python
#两个问题：自变量太多；补了部分-999，但仍存在缺失值
#结合题目要求，先去掉不重要的自变量sentFrameRate，sentBitrate，memory_app_used，memory_inactive；
#duration是累计值，与记录时刻时间节点有关，也去掉；
#要解决的问题是在工单提交时即判断，能用到的信息是工单提交前的观测，因此不考虑工单提交后的观测值
#剔除后剩余440个自变量
```


```python
#删掉不重要的变量
not_important = data.columns.str.contains('_lag_')+data.columns.str.contains('duration_')+data.columns.str.contains('sentFrameRate_')+data.columns.str.contains('sentBitrate_')+data.columns.str.contains('memory_app_used_')+data.columns.str.contains('memory_inactive_')
data_sample = data.loc[:,~not_important]
data_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>txAudioKBitrate_lead_1</th>
      <th>txAudioKBitrate_lead_2</th>
      <th>txAudioKBitrate_lead_3</th>
      <th>txAudioKBitrate_lead_4</th>
      <th>txAudioKBitrate_lead_5</th>
      <th>txAudioKBitrate_lead_6</th>
      <th>txAudioKBitrate_lead_7</th>
      <th>txAudioKBitrate_lead_8</th>
      <th>txAudioKBitrate_lead_9</th>
      <th>...</th>
      <th>memory_free_lead_31</th>
      <th>memory_free_lead_32</th>
      <th>memory_free_lead_33</th>
      <th>memory_free_lead_34</th>
      <th>memory_free_lead_35</th>
      <th>memory_free_lead_36</th>
      <th>memory_free_lead_37</th>
      <th>memory_free_lead_38</th>
      <th>memory_free_lead_39</th>
      <th>memory_free_lead_40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>54</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>...</td>
      <td>65912832.0</td>
      <td>57638912.0</td>
      <td>57344000.0</td>
      <td>57376768.0</td>
      <td>57196544.0</td>
      <td>56950784.0</td>
      <td>57540608.0</td>
      <td>58032128.0</td>
      <td>54214656.0</td>
      <td>52527104.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>54</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>53362688.0</td>
      <td>51200000.0</td>
      <td>48955392.0</td>
      <td>44285952.0</td>
      <td>39485440.0</td>
      <td>48824320.0</td>
      <td>33587200.0</td>
      <td>33882112.0</td>
      <td>35061760.0</td>
      <td>34324480.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>31195136.0</td>
      <td>31817728.0</td>
      <td>31080448.0</td>
      <td>28557312.0</td>
      <td>40370176.0</td>
      <td>40697856.0</td>
      <td>39944192.0</td>
      <td>40878080.0</td>
      <td>41385984.0</td>
      <td>37994496.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>53</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>54.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>80035840.0</td>
      <td>-999.0</td>
      <td>80658432.0</td>
      <td>-999.0</td>
      <td>81690624.0</td>
      <td>-999.0</td>
      <td>81379328.0</td>
      <td>-999.0</td>
      <td>79446016.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>53</td>
      <td>12.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>51.0</td>
      <td>43.0</td>
      <td>42.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 441 columns</p>
</div>




```python
#处理缺失值：当性能数据在日志服务器上缺失时，统一填入-999，故-999意为缺失值，替换为缺失值，一起处理。
#如缺失不多则扔掉
data_sample[data_sample == -999] = np.nan
data_sample.isnull().sum(axis = 0)
```

    D:\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    D:\Anaconda3\lib\site-packages\pandas\core\frame.py:3163: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._where(-key, value, inplace=True)
    




    type                        0
    txAudioKBitrate_lead_1      6
    txAudioKBitrate_lead_2      8
    txAudioKBitrate_lead_3      8
    txAudioKBitrate_lead_4      8
    txAudioKBitrate_lead_5      9
    txAudioKBitrate_lead_6      9
    txAudioKBitrate_lead_7      9
    txAudioKBitrate_lead_8     13
    txAudioKBitrate_lead_9     15
    txAudioKBitrate_lead_10    16
    txAudioKBitrate_lead_11    16
    txAudioKBitrate_lead_12    18
    txAudioKBitrate_lead_13    19
    txAudioKBitrate_lead_14    19
    txAudioKBitrate_lead_15    20
    txAudioKBitrate_lead_16    22
    txAudioKBitrate_lead_17    23
    txAudioKBitrate_lead_18    24
    txAudioKBitrate_lead_19    25
    txAudioKBitrate_lead_20    24
    txAudioKBitrate_lead_21    25
    txAudioKBitrate_lead_22    25
    txAudioKBitrate_lead_23    25
    txAudioKBitrate_lead_24    26
    txAudioKBitrate_lead_25    27
    txAudioKBitrate_lead_26    27
    txAudioKBitrate_lead_27    27
    txAudioKBitrate_lead_28    28
    txAudioKBitrate_lead_29    28
                               ..
    memory_free_lead_11        52
    memory_free_lead_12        54
    memory_free_lead_13        56
    memory_free_lead_14        61
    memory_free_lead_15        58
    memory_free_lead_16        62
    memory_free_lead_17        55
    memory_free_lead_18        67
    memory_free_lead_19        59
    memory_free_lead_20        69
    memory_free_lead_21        64
    memory_free_lead_22        69
    memory_free_lead_23        63
    memory_free_lead_24        69
    memory_free_lead_25        63
    memory_free_lead_26        71
    memory_free_lead_27        61
    memory_free_lead_28        72
    memory_free_lead_29        66
    memory_free_lead_30        73
    memory_free_lead_31        65
    memory_free_lead_32        74
    memory_free_lead_33        66
    memory_free_lead_34        70
    memory_free_lead_35        67
    memory_free_lead_36        69
    memory_free_lead_37        65
    memory_free_lead_38        73
    memory_free_lead_39        67
    memory_free_lead_40        71
    Length: 441, dtype: int64




```python
#可以尝试扔掉
data_notna = data_sample.dropna()
data_notna.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4892 entries, 0 to 4999
    Columns: 441 entries, type to memory_free_lead_40
    dtypes: float64(438), int64(3)
    memory usage: 16.5 MB
    


```python
data_notna.groupby('type').size()
#扔掉之后数量变化不大
```




    type
    0    3868
    1     378
    2     646
    dtype: int64




```python
data_notna.groupby('type').mean()
#初步看来，正常课堂和问题课堂在均值上还是有些差别的
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>txAudioKBitrate_lead_1</th>
      <th>txAudioKBitrate_lead_2</th>
      <th>txAudioKBitrate_lead_3</th>
      <th>txAudioKBitrate_lead_4</th>
      <th>txAudioKBitrate_lead_5</th>
      <th>txAudioKBitrate_lead_6</th>
      <th>txAudioKBitrate_lead_7</th>
      <th>txAudioKBitrate_lead_8</th>
      <th>txAudioKBitrate_lead_9</th>
      <th>txAudioKBitrate_lead_10</th>
      <th>...</th>
      <th>memory_free_lead_31</th>
      <th>memory_free_lead_32</th>
      <th>memory_free_lead_33</th>
      <th>memory_free_lead_34</th>
      <th>memory_free_lead_35</th>
      <th>memory_free_lead_36</th>
      <th>memory_free_lead_37</th>
      <th>memory_free_lead_38</th>
      <th>memory_free_lead_39</th>
      <th>memory_free_lead_40</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.876939</td>
      <td>50.954757</td>
      <td>50.850569</td>
      <td>50.889607</td>
      <td>50.907187</td>
      <td>50.976732</td>
      <td>51.031282</td>
      <td>51.080920</td>
      <td>50.999741</td>
      <td>51.044726</td>
      <td>...</td>
      <td>4.765132e+07</td>
      <td>4.778104e+07</td>
      <td>4.792880e+07</td>
      <td>4.783270e+07</td>
      <td>4.808562e+07</td>
      <td>4.815841e+07</td>
      <td>4.800079e+07</td>
      <td>4.830218e+07</td>
      <td>4.829248e+07</td>
      <td>4.831507e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.587302</td>
      <td>44.838624</td>
      <td>46.052910</td>
      <td>47.161376</td>
      <td>48.489418</td>
      <td>49.433862</td>
      <td>49.423280</td>
      <td>49.756614</td>
      <td>49.933862</td>
      <td>49.791005</td>
      <td>...</td>
      <td>5.211688e+07</td>
      <td>5.213331e+07</td>
      <td>5.198894e+07</td>
      <td>5.227487e+07</td>
      <td>5.247369e+07</td>
      <td>5.234371e+07</td>
      <td>5.267153e+07</td>
      <td>5.238125e+07</td>
      <td>5.241742e+07</td>
      <td>5.222883e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45.057276</td>
      <td>46.164087</td>
      <td>46.902477</td>
      <td>48.320433</td>
      <td>48.856037</td>
      <td>49.448916</td>
      <td>49.287926</td>
      <td>49.555728</td>
      <td>49.834365</td>
      <td>49.696594</td>
      <td>...</td>
      <td>5.882427e+07</td>
      <td>5.819966e+07</td>
      <td>5.815969e+07</td>
      <td>5.818918e+07</td>
      <td>5.777626e+07</td>
      <td>5.791880e+07</td>
      <td>5.787236e+07</td>
      <td>5.802911e+07</td>
      <td>5.792340e+07</td>
      <td>5.790435e+07</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 440 columns</p>
</div>




```python

```


```python
#考虑使用逻辑回归进行分类，尝试把变量都放进去
X = data_notna.drop(['type'],axis = 1)
y = data_notna['type']

#自动设置训练样本与测试样本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
#将数据归一化
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)
```


```python
#建模
lr = LR(penalty = 'l1',class_weight = 'balanced') #模型特征比较多，为了让系数稀疏化，使用L1正则化；同时调节样本类型不平衡问题
lr.fit(X_train_std,y_train)
#训练结果
pred_train = lr.predict(X_train_std)
precision_train = precision_score(y_train, pred_train, average='weighted')
recall_train = recall_score(y_train, pred_train, average='weighted')
f1_train = f1_score(y_train, pred_train, average='weighted')
report_train = classification_report(y_train, pred_train)
cm_train = confusion_matrix(y_train, pred_train)
#测试结果
pred_test = lr.predict(X_test_std)
precision_test = precision_score(y_test, pred_test, average='weighted')
recall_test = recall_score(y_test, pred_test, average='weighted')
f1_test = f1_score(y_test, pred_test, average='weighted')
report_test = classification_report(y_test, pred_test)
cm_test = confusion_matrix(y_test, pred_test)
```


```python
#print('训练集精确率：',precision_train)
#print('训练集召回率：',recall_train)
#print('训练集f1值：',f1_train)
print('训练集分类报告：\n',report_train)
print('训练集混淆矩阵：\n',cm_train)
```

    训练集分类报告：
                  precision    recall  f1-score   support
    
              0       0.94      0.95      0.95      3089
              1       0.62      0.55      0.58       293
              2       0.72      0.70      0.71       531
    
    avg / total       0.89      0.89      0.89      3913
    
    训练集混淆矩阵：
     [[2949   51   89]
     [  76  161   56]
     [ 109   49  373]]
    


```python
#print('测试集精确率：',precision_test)
#print('测试集召回率：',recall_test)
#print('测试集f1值：',f1_test)
print('测试集分类报告：\n',report_test)
print('测试集混淆矩阵：\n',cm_test)
```

    测试集分类报告：
                  precision    recall  f1-score   support
    
              0       0.92      0.94      0.93       779
              1       0.37      0.29      0.33        85
              2       0.56      0.57      0.56       115
    
    avg / total       0.83      0.84      0.84       979
    
    测试集混淆矩阵：
     [[735  22  22]
     [ 31  25  29]
     [ 30  20  65]]
    


```python
#测试集分类概率
pred = lr.predict_proba(X_test_std)
result = pd.DataFrame(pred)
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000945</td>
      <td>0.488422</td>
      <td>0.510633</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.059358</td>
      <td>0.147527</td>
      <td>0.793115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.529422</td>
      <td>0.423660</td>
      <td>0.046918</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.849420</td>
      <td>0.094632</td>
      <td>0.055948</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.921200</td>
      <td>0.070245</td>
      <td>0.008555</td>
    </tr>
  </tbody>
</table>
</div>




```python
#如果把正常课堂去掉，只用1,2样本建模，结果不是很理想
```


```python

```


```python
#试了下xgboost方法，时间所迫，弃之
model = xgb.XGBClassifier(max_depth=3,n_estimators=200,learn_rate=0.01)
model.fit(X_train,y_train)
test_pred = model.predict(X_test)
xgb_precision = precision_score(y_test,test_pred,average='weighted')
xgb_report = classification_report(y_test,test_pred)
print(xgb_precision)
print(xgb_report)
```

    0.8263868449822172
                 precision    recall  f1-score   support
    
              0       0.91      0.98      0.94       779
              1       0.45      0.11      0.17        85
              2       0.56      0.57      0.57       115
    
    avg / total       0.83      0.86      0.83       979
    
    

    D:\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    


```python

```
