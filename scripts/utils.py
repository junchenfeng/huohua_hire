import os
import pickle
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize, scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample as hp_sample

SEED = 43
BASE_VAR = ['txAudioKBitrate', 'rxAudioKBitrate', 'txVideoKBitrate', 'rxVideoKBitrate', 'duration', 'userCount',   
            'cpuTotalUsage', 'cpuAppUsage', 'sentFrameRate', 'sentBitrate', 'memory_app_used', 'memory_free', 
            'cpu', 'lag', 'fps', 'memory_inactive']

# 整理变量字典
def feature_range(x):
    """
    Args:
    ----------------
    x: pd.DataFrame

    Returns:
    ----------------
    feature_dict_df: pd.DataFrame
    """
    feature_dict_df = pd.DataFrame(x.dtypes,columns=['dtypes'])
    feature_dict_df = feature_dict_df.reset_index()
    feature_dict_df['var_code'] = feature_dict_df['index']
    feature_dict_df = feature_dict_df[['var_code','dtypes']]
    feature_dict_df['missing'] = x.isnull().sum().values  
    feature_dict_df['uniques'] = x.nunique().values
    feature_dict_df['first_value'] = x.iloc[0].values
    feature_dict_df['second_value'] = x.iloc[1].values
    feature_dict_df['third_value'] = x.iloc[2].values

    for name in feature_dict_df['var_code'].value_counts().index:
        feature_dict_df.loc[feature_dict_df['var_code'] == name, 'entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2) 

    return feature_dict_df

def eda(x, var_dict, unique_cutoff=0.97, is_all=True, save_label=None, save=False, **paths):
    """
    生成一份特征summary文件

    Args:
    ----------------
    x(pd.DataFrame): 数据集
    var_dict(pd.DataFrame): 变量字典, 必须包括var_code, data_source, chinese_var, value_type, var_type等列
    data_path(str): 本函数最后生成一份summary文件, 该文件存储路径
    save_label(str): 文件名 'feature_summary_%s.xlsx' % (save_label)
    unique_cutoff(fraction): 单变量剔除时的阈值
    is_all(boolean): True, 全量样本统计数据, summay不统计excluded variable; 
        False, 根据unique_cutoff更新exclusion_reason列.

    Returns:
    ----------------
    None
    """
    if paths:
        data_path = paths['data_path']

    x_ = x.copy()
    if x_ is None:
        print('X数据集未赋值')
        return
    if var_dict is None:
        print('var_dict为空, 请确认var_dict可用')
        return
    print('warning: X 类型为 {}, var_dict 类型为 {}'.format(
        x_.__class__, var_dict.__class__))

    feature_summary = var_dict['var_code'].to_frame('var_code')
    feature_summary.index = feature_summary['var_code']
    feature_summary.drop('var_code', axis=1, inplace=True)
    feature_summary['N'] = x_.shape[0]

    feature_summary['N_missing'] = x_[var_dict['var_code']].isnull().sum()
    feature_summary['pct_missing'] = np.round(x_[var_dict['var_code']].isnull().sum() * 1.0/x_.shape[0], 3)
    feature_summary['N_-999'] = (x_[var_dict['var_code']] == -999).sum()
    feature_summary['pct_-999'] = np.round((x_[var_dict['var_code']] == -999).sum() * 1.0/x_.shape[0], 3)
    feature_summary['N_NA'] = feature_summary['N_missing'] + feature_summary['N_-999']
    feature_summary['pct_NA'] = np.round(feature_summary['N_NA']/x_.shape[0], 3)
    feature_summary['N_0'] = (x_[var_dict['var_code']] == 0).sum()
    feature_summary['pct_0'] = np.round(feature_summary['N_0'] * 1.0/x_.shape[0], 3)

    feature_summary.loc[:, 'mean'] = x_[var_dict['var_code']].replace(-999, np.nan).mean().round(1)
    feature_summary.loc[:, 'std'] = x_[var_dict['var_code']].replace(-999, np.nan).std().round(1)
    feature_summary.loc[:, 'min'] = x_[var_dict['var_code']].replace(-999, np.nan).min().round(1)
    feature_summary.loc[:, 'median'] = x_[var_dict['var_code']].replace(-999, np.nan).median().round(1)
    feature_summary.loc[:, 'max'] = x_[var_dict['var_code']].replace(-999, np.nan).max().round(1)
    feature_summary.loc[:, 'p01'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(0.01)
    feature_summary.loc[:, 'p05'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.05)
    feature_summary.loc[:, 'p10'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.10)
    feature_summary.loc[:, 'p15'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.15)
    feature_summary.loc[:, 'p25'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.25)
    feature_summary.loc[:, 'p75'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.75)
    feature_summary.loc[:, 'p90'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.90)
    feature_summary.loc[:, 'p95'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.95)
    feature_summary.loc[:, 'p99'] = x_[var_dict['var_code']].replace(-999, np.nan).quantile(q=0.99)
    # reorder columns
    if is_all:
        col_reorder = ['N','N_missing','pct_missing','N_-999','pct_-999','N_NA','pct_NA', 
                'N_0','pct_0','min','mean','median','max','std','p01','p05','p10', 
                'p15','p25','p75','p90','p95','p99']
    else:
        col_reorder = ['N','exclusion_reason','N_missing','pct_missing','N_-999','pct_-999', 
                'N_NA','pct_NA','N_0','pct_0','min','mean','median','max','std', 'p01', 
                'p05','p10','p15','p25','p75','p90','p95','p99']
        feature_summary.loc[feature_summary['pct_0' > unique_cutoff],'exclusion_reason'] = '0值占比>{}'.format(unique_cutoff)
        feature_summary.loc[feature_summary['pct_NA' > unique_cutoff],'exclusion_reason'] = 'NA占比>{}'.format(unique_cutoff)
    
    feature_summary = feature_summary[col_reorder]
    if save_label is None:
        save_label = 'all'
        print('info: save_label未输入, 文件名默认为feature_summary_all.xlsx')

    file_name = 'feature_summary_{}.xlsx'.format(save_label)

    if save:
        if data_path is None:
            feature_summary.to_excel(file_name)
        else:
            feature_summary.to_excel(os.path.join(data_path, file_name))
        print('info: {}已保存'.format(file_name))

def save_data_to_pickle(obj, file_path, file_name):
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'wb') as outfile:
        pickle.dump(obj, outfile)
    print('info: file save finished!')

def load_data_from_pickle(file_path, file_name):
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'rb') as infile:
        result = pickle.load(infile)
    return result

def plot_missing_dist(base_var, data, with_default=True, target_column='type', save=False):
    """
    plot missing value counts from last lead to last lag for each feature in type [0,1,2].

    Args:
    ----------------
    base_var(list): feature name list
    data(pd.DataFrame): sample data set
    with_default(boolean): True for -999, False for NAN

    Returns:
    ----------------
    None
    """
    xs = np.linspace(-40, 40, 82, dtype=int)
    xs = np.hstack((xs[xs < 0], xs[xs > 0]))
    cmap = plt.cm.jet
    cols = 4
    rows = len(base_var)//cols
    fig, axes = plt.subplots(rows,cols,figsize=(15,16))
    pic_title = '-999_counts_trend' if with_default else 'NAN_counts_trend'
    save_label = '-999_value_counts.png' if with_default else 'NAN_value_counts.png'
    for idx, tmp_var in enumerate(base_var):
        tmp_var_lead = [tmp_var+'_{}_{}'.format('lead', i+1) for i in range(40)]
        tmp_var_lead.reverse()
        tmp_var_lag = [tmp_var+'_{}_{}'.format('lag', i+1) for i in range(40)]
        tmp_vars = tmp_var_lead + tmp_var_lag + [target_column]
        tmp_data = data[tmp_vars]

        if with_default:
            y0 = list((tmp_data[tmp_data.type == 0].drop(target_column, 1) == -999).sum())
            y1 = list((tmp_data[tmp_data.type == 1].drop(target_column, 1) == -999).sum())
            y2 = list((tmp_data[tmp_data.type == 2].drop(target_column, 1) == -999).sum())
        else:
            y0 = list(tmp_data[tmp_data.type == 0].drop(target_column, 1).isnull().sum())
            y1 = list(tmp_data[tmp_data.type == 1].drop(target_column, 1).isnull().sum())
            y2 = list(tmp_data[tmp_data.type == 2].drop(target_column, 1).isnull().sum())

        axes[idx//cols][idx%cols].plot(xs, y0, c=cmap(0./3), label='type0')
        axes[idx//cols][idx%cols].plot(xs, y1, c=cmap(1./3), label='type1')
        axes[idx//cols][idx%cols].plot(xs, y2, c=cmap(2./3), label='type2')
        axes[idx//cols][idx%cols].set_xlabel(tmp_var)
        axes[idx//cols][idx%cols].set_ylabel('missing_value_counts')
        axes[idx//cols][idx%cols].legend()
    # plt.legend()
    plt.suptitle(pic_title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95,bottom=0.05,wspace=0.2,hspace=0.2)
    if save:
        plt.savefig('images/3_{}'.format(save_label))
    plt.close()
        
def generate_feature(x, var_dict, features):
    x_ = x.copy()
    x_ = x_.astype('float')
    for _, ele in enumerate(features):
        lead_ele, lag_ele = ele+'_lead', ele+'_lag'
        print('{}{}:'.format(ele, '_lead'), np.sum(var_dict['var_code'].apply(lambda tmp: lead_ele in tmp)))
        print('{}{}:'.format(ele, '_lag'), np.sum(var_dict['var_code'].apply(lambda tmp: lag_ele in tmp)))
        lead_tmp = x_[var_dict[var_dict['var_code'].apply(lambda tmp: lead_ele in tmp)]['var_code']]
        lag_tmp = x_[var_dict[var_dict['var_code'].apply(lambda tmp: lag_ele in tmp)]['var_code']]
        total_tmp = pd.concat([lead_tmp, lag_tmp],axis=1)

        x_['{}_total_min'.format(ele)] = total_tmp.min(axis=1).round(2)
        x_['{}_total_max'.format(ele)] = total_tmp.max(axis=1).round(2)
        x_['{}_total_mean'.format(ele)] = total_tmp.mean(axis=1).round(2)
        x_['{}_total_sum'.format(ele)] = total_tmp.sum(axis=1).round(2)
        x_['{}_total_mode'.format(ele)] = total_tmp.mode(axis=1)[0]
        x_['{}_total_p01'.format(ele)] = total_tmp.quantile(q=0.01,axis=1).round(2)
        x_['{}_total_p05'.format(ele)] = total_tmp.quantile(q=0.05,axis=1).round(2)
        x_['{}_total_p10'.format(ele)] = total_tmp.quantile(q=0.10,axis=1).round(2)
        x_['{}_total_p25'.format(ele)] = total_tmp.quantile(q=0.25,axis=1).round(2)
        x_['{}_total_p50'.format(ele)] = total_tmp.quantile(q=0.50,axis=1).round(2)
        x_['{}_total_p75'.format(ele)] = total_tmp.quantile(q=0.75,axis=1).round(2)
        x_['{}_total_p90'.format(ele)] = total_tmp.quantile(q=0.90,axis=1).round(2)
        x_['{}_total_p95'.format(ele)] = total_tmp.quantile(q=0.95,axis=1).round(2)
        x_['{}_total_p99'.format(ele)] = total_tmp.quantile(q=0.99,axis=1).round(2)

        x_['{}_{}'.format(ele, 'lead_min')] = lead_tmp.min(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_mean')] = lead_tmp.mean(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_max')] = lead_tmp.max(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_sum')] = lead_tmp.sum(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_mode')] = lead_tmp.mode(axis=1)[0]
        x_['{}_{}'.format(ele, 'lead_p01')] = lead_tmp.quantile(q=0.01, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p05')] = lead_tmp.quantile(q=0.05, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p10')] = lead_tmp.quantile(q=0.10, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p25')] = lead_tmp.quantile(q=0.25, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p50')] = lead_tmp.quantile(q=0.50, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p75')] = lead_tmp.quantile(q=0.75, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p90')] = lead_tmp.quantile(q=0.90, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p95')] = lead_tmp.quantile(q=0.95, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lead_p99')] = lead_tmp.quantile(q=0.99, axis=1).round(2)

        x_['{}_{}'.format(ele, 'lag_min')] = lag_tmp.min(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_mean')] = lag_tmp.mean(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_max')] = lag_tmp.max(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_sum')] = lag_tmp.sum(axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_mode')] = lag_tmp.mode(axis=1)[0]
        x_['{}_{}'.format(ele, 'lag_p01')] = lag_tmp.quantile(q=0.01, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p05')] = lag_tmp.quantile(q=0.05, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p10')] = lag_tmp.quantile(q=0.10, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p25')] = lag_tmp.quantile(q=0.25, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p50')] = lag_tmp.quantile(q=0.50, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p75')] = lag_tmp.quantile(q=0.75, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p90')] = lag_tmp.quantile(q=0.90, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p95')] = lag_tmp.quantile(q=0.95, axis=1).round(2)
        x_['{}_{}'.format(ele, 'lag_p99')] = lag_tmp.quantile(q=0.99, axis=1).round(2)

        x_['{}_{}'.format(ele, 'lead_lag_min_diff')] = x_['{}_{}'.format(ele, 'lead_min')]-x_['{}_{}'.format(ele, 'lag_min')]
        x_['{}_{}'.format(ele, 'lead_lag_max_diff')] = x_['{}_{}'.format(ele, 'lead_max')]-x_['{}_{}'.format(ele, 'lag_max')]
        x_['{}_{}'.format(ele, 'lead_lag_sum_diff')] = x_['{}_{}'.format(ele, 'lead_sum')]-x_['{}_{}'.format(ele, 'lag_sum')]
        x_['{}_{}'.format(ele, 'lead_lag_mean_diff')] = x_['{}_{}'.format(ele, 'lead_mean')]-x_['{}_{}'.format(ele, 'lag_mean')]
        x_['{}_{}'.format(ele, 'lead_lag_p01_diff')] = x_['{}_{}'.format(ele, 'lead_p01')]-x_['{}_{}'.format(ele, 'lag_p01')]
        x_['{}_{}'.format(ele, 'lead_lag_p05_diff')] = x_['{}_{}'.format(ele, 'lead_p05')]-x_['{}_{}'.format(ele, 'lag_p05')]
        x_['{}_{}'.format(ele, 'lead_lag_p10_diff')] = x_['{}_{}'.format(ele, 'lead_p10')]-x_['{}_{}'.format(ele, 'lag_p10')]
        x_['{}_{}'.format(ele, 'lead_lag_p25_diff')] = x_['{}_{}'.format(ele, 'lead_p25')]-x_['{}_{}'.format(ele, 'lag_p25')]
        x_['{}_{}'.format(ele, 'lead_lag_p50_diff')] = x_['{}_{}'.format(ele, 'lead_p50')]-x_['{}_{}'.format(ele, 'lag_p50')]
        x_['{}_{}'.format(ele, 'lead_lag_p75_diff')] = x_['{}_{}'.format(ele, 'lead_p75')]-x_['{}_{}'.format(ele, 'lag_p75')]
        x_['{}_{}'.format(ele, 'lead_lag_p90_diff')] = x_['{}_{}'.format(ele, 'lead_p90')]-x_['{}_{}'.format(ele, 'lag_p90')]
        x_['{}_{}'.format(ele, 'lead_lag_p95_diff')] = x_['{}_{}'.format(ele, 'lead_p95')]-x_['{}_{}'.format(ele, 'lag_p95')]
        x_['{}_{}'.format(ele, 'lead_lag_p99_diff')] = x_['{}_{}'.format(ele, 'lead_p99')]-x_['{}_{}'.format(ele, 'lag_p99')]
    return x_

HP_SPACE = {
    'clf_type': hp.choice('clf_type', [
        {
            'model': 'svm',
            'clf': {
                'C': hp.uniform('C', 0, 20)
                , 'gamma': hp.uniform('svm.gamma', 0, 20)
                , 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf', 'poly'])
                , 'decision_function_shape': hp.choice('decision_function_shape', ['ovr', 'ovo'])
                , 'class_weight': hp.choice('svm.class_weight', [None, 'balanced'])
                , 'scale': hp.choice('svm.scale', [0, 1])
                , 'normalize': hp.choice('svm.normalize', [0, 1])
            }
        },
        {
            'model': 'rf',
            'clf': {
                'max_depth': hp.choice('rf.max_depth', range(2, 5))
                , 'max_features': hp.uniform('max_features', low=0.25, high=0.75)
                , 'n_estimators': hp.choice('rf.n_estimators', range(5, 20))
                , 'criterion': hp.choice('criterion', ["gini", "entropy"])
                , 'class_weight': hp.choice('rf.class_weight', [None, 'balanced'])
                , 'scale': hp.choice('rf.scale', [0, 1])
                , 'normalize': hp.choice('rf.normalize', [0, 1])
            }
        },
        {
            'model':'xgb',
            'clf': {
                'max_depth': hp.choice('xgb.max_depth', [4,5,6])
                , 'booster': hp.choice('booster', ['gblinear', 'gbtree', 'dart'])
                , 'n_estimators': hp.randint('xgb.n_estimators', 50) 
                , 'min_child_weight': hp.randint('min_child_weight', 10)
                , 'gamma': hp.uniform('gamma', 0.001, 0.1)
                , 'subsample': hp.uniform('subsample', 0.1, 0.7)
                , 'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.7)
                , 'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 0.7)
                , 'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 0.7)
                , 'learning_rate': hp.uniform('learning_rate', 0.001, 0.2)
                , 'reg_lambda': hp.randint('reg_lambda', 4)
                , 'eval_metric': hp.choice('eval_metric', ['mlogloss', 'merror'])
                , 'scale_pos_weight': hp.randint('scale_pos_weight', 3)
                , 'scale': hp.choice('xgb.scale', [0, 1])
                , 'normalize': hp.choice('xgb.normalize', [0, 1])
            }
        }
    ])
}

def draw_hp_sample(hps=HP_SPACE):
    """Draw random sample to see if hyperspace is correctly defined"""
    print(hp_sample(hps))

def opt_func(hyspace, x, y, ncv):
    """
    Target function for optimization

    Args:
    ----------------
    hpspace : sample point from search space
    x : feature matrix
    y : target array
    cv : int/StratifiedKFold

    Returns:
    ----------------
    dict(
        'loss' : target function value (negative mean cross-validation f1_score)
        'cv_std' : cross-validation f1_score standard deviation
        'status' : status of function evaluation
    )
    """

    model, model_params = hyspace['clf_type']['model'], hyspace['clf_type']['clf']

    x_ = x.copy()
    if 'normalize' in model_params:
        if model_params['normalize'] == 1:
            x_ = normalize(x_)
        del model_params['normalize']
    if 'scale' in model_params:
        if model_params['scale'] == 1:
            x_ = scale(x_)
        del model_params['scale']

    if model == 'svm':
        clf = SVC(**model_params)
        score = cross_val_score(clf, x_, y, scoring='f1_macro', cv=StratifiedKFold(ncv, True, random_state = SEED))
    elif model == 'rf':
        clf = RandomForestClassifier(**model_params)
        score = cross_val_score(clf, x_, y, scoring='f1_macro', cv=StratifiedKFold(ncv, True, random_state = SEED))
    elif model == 'dt':
        clf = DecisionTreeClassifier(**model_params)
        score = cross_val_score(clf, x_, y, scoring='f1_macro', cv=StratifiedKFold(ncv, True, random_state = SEED))
    elif model == 'xgb':
        model_params['n_estimators'] = model_params['n_estimators'] + 1
        model_params['min_child_weight'] = model_params['min_child_weight'] + 1
        model_params['reg_lambda'] = model_params['reg_lambda'] + 1
        model_params['verbosity'] = 2
        model_params['objective'] = 'multi:softmax'
        model_params['eval_metric']= 'mlogloss'
        model_params['num_class'] = 3
        clf = xgb.XGBClassifier(**model_params)
        score = cross_val_score(clf, x_, y, cv=StratifiedKFold(ncv, True, random_state = SEED))
        score = -1 * score
    return {'loss': -score.mean(), 'std': score.std(), 'status': STATUS_OK}

def hyper_best(x_train, y_train, f=opt_func, space=HP_SPACE, algo=tpe.suggest, max_evals=100, ncv=5):
    trials = Trials()
    best = fmin(partial(opt_func, x=x_train, y=y_train, ncv=ncv), space=space, algo=algo, max_evals=max_evals, trials=trials)
    print('best:\n', best)
    return best, trials

def plot_best_param(rn, cn, parameters, trials, save=False):
    """plot model loss of parameter over each iteration.

    Args:
    ----------------
    rn(int): rows of ax
    cn(int): cols of ax
    parameter(dict): paramter dictionary
    trials(Trials object like a dictionary): include iteration times and 
        loss values of each iter.

    Returns:
    ----------------
    None
    """
    f, ax = plt.subplots(nrows=rn, ncols=cn, figsize=(20, 5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        ax[i//cn][i%cn].scatter(xs, ys, s=20,c=cmap(float(i)/len(parameters)))
        ax[i//cn][i%cn].set_title(val)
        ax[i//cn][i%cn].set_ylabel('mlogloss')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95,bottom=0.05,wspace=0.2,hspace=0.4)
    if save:
        plt.savefig('images/loss_parameter.png')
    plt.show()
    
def plot_logloss(model, save=False):
    results = model.evals_result_
    epochs = len(results['validation_0']['mlogloss'])
    xs = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(xs, results['validation_0']['mlogloss'], label='Train')
    ax.plot(xs, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('MLogloss')
    plt.title('Train Test Logloss Curve')
    if save:
        plt.savefig('images/mlogloss_curve.png')
    plt.show()
    
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    print('original data size {:.2f}'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df