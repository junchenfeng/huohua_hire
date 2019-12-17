import sys
import warnings
warnings.filterwarnings('once')
sys.path.append('scripts') 

import utils
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from imp import reload
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(['pwd']).decode('utf8'))


# load data
raw_data = pd.read_csv('data/data.csv')
round(raw_data.memory_usage().sum()/1024**2,2) #48MB
np.random.seed(0)

"""
Taka a look at the whole data
1. 5000个样本, 1281个变量
2. 数据基本是int, 除了cpu有负值-1, 其他有缺失值-999, 及nan以外, 数据基本都为连续的正数
3. 有些列的值为-999, 占总数据量的0.21%
4. 有些列有missing value, 占总数据量的0.14%
5. 目标列type有三个值: 0--视频流畅, 1---网络抖动, 2---需要提交给技术支持工单
6. 从各类别-999与NAN的分布来看, 二者分布相同, 可以考虑将NAN都填充为-999
"""
raw_data.shape #(5000, 1281)
raw_data.sample(5) 

sns.countplot(x=raw_data['type'],label='count')
plt.savefig('images/1_type_value_counts.png')
plt.show()
type0, type2, type1 = raw_data['type'].value_counts() #3944, 668, 388
pd.crosstab(raw_data['type'],'count')/raw_data.shape[0] #(0,0.7888),(1,0.0776),(2,0.1336)
# 整理变量字典
reload(utils)
var_dict = utils.feature_range(raw_data)
var_dict.sample(5)
utils.eda(raw_data, var_dict, save_label='P1', data_path='result_files')
feature_dict = pd.read_excel('result_files/feature_summary_P1.xlsx')
feature_dict.sample(5)
var_dict = pd.read_excel('result_files/feature_dict.xlsx')
var_dict.sample(5)

# 看看-999的占比: 0.21%
default_mis_df = (raw_data == -999).sum()
default_mis_count = default_mis_df.sum()
default_mis_count
default_mis_pct = (default_mis_count/np.product(raw_data.shape)) * 100
default_mis_pct # 0.21%

type0_default_mis = (raw_data[raw_data.type == 0] == -999).sum().sum()
type1_default_mis = (raw_data[raw_data.type == 1] == -999).sum().sum()
type2_default_mis = (raw_data[raw_data.type == 2] == -999).sum().sum()
type0_default_mis_pct = (type0_default_mis/np.product(raw_data.shape)) * 100 # type0_-999: 0.14%
type1_default_mis_pct = (type1_default_mis/np.product(raw_data.shape)) * 100 # type1_-999: 0.02%
type2_default_mis_pct = (type2_default_mis/np.product(raw_data.shape)) * 100 # type2_-999: 0.05%

# 看看nan的占比: 0.14%
nan_mis_df = raw_data.isnull().sum()
nan_mis_df_count = nan_mis_df.sum()
nan_mis_df_count
nan_mis_pct = (nan_mis_df_count/np.product(raw_data.shape))*100
nan_mis_pct # 0.14%
type0_mis = raw_data[raw_data.type == 0].isnull().sum().sum()
type1_mis = raw_data[raw_data.type == 1].isnull().sum().sum()
type2_mis = raw_data[raw_data.type == 2].isnull().sum().sum()
type0_mis_pct = (type0_mis/np.product(raw_data.shape)) * 100 # type0_nan: 0.07%
type1_mis_pct = (type1_mis/np.product(raw_data.shape)) * 100 # type1_nan: 0.02%
type2_mis_pct = (type2_mis/np.product(raw_data.shape)) * 100 # type2_nan: 0.04%

# NAN_COUNTS与-999_COUNTS分布图
# plt.figure(figsize=(150,150))
ax2 = plt.subplot2grid((2,2),(0,0))
ax3 = plt.subplot2grid((2,2),(0,1))
ax1 = plt.subplot2grid((2,2),(1,0),colspan=2)
# 各类别nan_counts与nan_pct分布图
x2 = np.linspace(0, 2, 3, dtype=int)
ax2.bar(x2, [type0_mis, type1_mis, type2_mis],color='tab:blue')
ax2.set_ylabel('count_of_nan')
ax2.set_xticks(x2)
ax2.set_xticklabels(['type0','type1','type2'])
ax2_2 = ax2.twinx()
ax2_2.plot(x2, np.array([type0_mis_pct, type1_mis_pct, type2_mis_pct]).astype('float16').round(2), '-', color='red')
ax2_2.set_ylabel('pct_of_nan')
ax2_2.set_ylim(0, 0.15)
# 各类别-999_counts与-999_pct分布图
ax3.bar(x2, [type0_default_mis, type1_default_mis, type2_default_mis],color='tab:blue')
ax3.set_ylabel('count_of_-999')
ax3.set_xticks(x2)
ax3.set_xticklabels(['type0','type1','type2'])
ax3_2 = ax3.twinx()
ax3_2.plot(x2, np.array([type0_default_mis_pct, type1_default_mis_pct, type2_default_mis_pct],dtype='float16').round(2), '-', color='red')
ax3_2.set_ylabel('pct_of_-999')
ax3_2.set_ylim(0, 0.2)
# all
x1 = np.linspace(1, 2, 2, dtype=int)
ax1.bar(x1, [default_mis_count,nan_mis_df_count],color='tab:blue')
ax1.set_ylabel('count')
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['-999', 'NAN'])
ax1_2 = ax1.twinx()
ax1_2.plot(x1, np.array([default_mis_pct, nan_mis_pct],dtype='float16').round(2),color='red')
ax1_2.set_ylabel('pct')
plt.suptitle('Missing_Value_Counts')
plt.subplots_adjust(wspace=0.9, hspace=0.2)
plt.savefig('images/2_Missing_Value_Counts.png')
plt.show()

"""
1. 观察整个数据集正常数据范围为[0, Infinity].
2. 对变量进行处理, 将每个变量衍出一个是否有缺失值的变量
3. 进一步观察缺失值与类别的关系, 如与类别相关性低, 可以考虑移除这部分样本.
"""

base_var = copy.deepcopy(utils.BASE_VAR)
reload(utils)
utils.plot_missing_dist(base_var, raw_data.replace(-999,np.nan), False)

# 填充缺失值
all_wom_data = raw_data.fillna(-999)

# 数据集类型转换
all_wom_data = all_wom_data.astype('float')
all_wom_data['type'] = all_wom_data['type'].astype('int')
all_wom_data = utils.reduce_mem_usage(all_wom_data) #48.87Mb->14.50 Mb
utils.save_data_to_pickle(all_wom_data, 'data', 'all_wom_data.pkl')

all_wom_data['type']

# 观察单变量80次监控数据分布
# 去掉缺失样本后的16个特征80次监控日志数据趋势分布
remove_missing_data = all_wom_data.replace(-999, np.nan).dropna()
remove_missing_data.shape

x = np.linspace(-40, 40, 82, dtype=int)
x = np.hstack((x[x < 0], x[x > 0]))
cmap = plt.cm.jet
cols = 4
rows = 4
fig, axes = plt.subplots(rows,cols,figsize=(15,16))
for idx, tmp_var in enumerate(base_var):
    tmp_var_lead = [tmp_var+'_{}_{}'.format('lead', i+1) for i in range(40)]
    tmp_var_lead.reverse()
    tmp_var_lag = [tmp_var+'_{}_{}'.format('lag', i+1) for i in range(40)]
    tmp_vars = tmp_var_lead + tmp_var_lag
    y_mean = remove_missing_data.mean()[tmp_vars].values
    y_min = remove_missing_data.min()[tmp_vars].values
    y_max = remove_missing_data.max()[tmp_vars].values
    axes[idx//cols][idx%cols].plot(x, y_min, c=cmap(0./3), label='min')
    axes[idx//cols][idx%cols].plot(x, y_mean, c=cmap(1./3), label='mean')
    axes[idx//cols][idx%cols].plot(x, y_max, c=cmap(2./3), label='max')
    axes[idx//cols][idx%cols].set_ylabel(tmp_var)
    axes[idx//cols][idx%cols].legend()
plt.suptitle('Wom_Sample_Distribution')
plt.tight_layout()
plt.subplots_adjust(top=0.95,bottom=0.03,hspace=0.3,wspace=0.3)
plt.savefig('images/7_wom_sample_distribution.png')
plt.show()