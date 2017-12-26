
# coding: utf-8

# # Recruit restaurant competition

# ## Some thoughts about tasks
# * Convert dates and times
# * Summarise tables: one row per restaurant and day
# * Join with store info
# * Join with actual air visit data
# * Join with holidays data
# * Think about what to do with hpg data...
# 
# Remember any features extracted from training tables need to be extractable from submission table.
# 
# Remember that we are predicting visits, not reservations.
# 
# Check if we have reservation data for some or all submission dates.
#  - We have reservation data for test and training data.
# 
# Check if the restaurants in the submission list are all included in the training data.
#
# Latitude and logitude are highly correlated: can we combine to a single feature?

# ## Loading modules

import pandas as pd
import sklearn as sk

# ## Loading raw data

air_reserve = pd.read_csv('input/air_reserve.csv')
air_store_info = pd.read_csv('input/air_store_info.csv')
air_visit_data = pd.read_csv('input/air_visit_data.csv')
hpg_reserve = pd.read_csv('input/hpg_reserve.csv')
hpg_store_info = pd.read_csv('input/hpg_store_info.csv')
date_info = pd.read_csv('input/date_info.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')
store_id_relation = pd.read_csv('input/store_id_relation.csv')

#%%
# Checking submission samples
# What's interesting is that predictions are only required for places in the "air" booking system. Thus, how should we use HPG infos?

sample_submission.sort_values('id').tail()

#%%
# A look at date_info table
# Good news: seems like we get the holiday info for every date in the submission table.

date_info.tail()

#%%
# Extract info from dates
# First, define a little function to extract year, month, day, hour

def extract_dates(pd_df, target_var, format_str="%Y-%m-%d %H:%M:%S", prefix=None):
    if not prefix:
        prefix = target_var
    pd_df[target_var] = pd.to_datetime(pd_df[target_var], format=format_str)
    pd_df['{0}_year'.format(prefix)]  = pd.DatetimeIndex(pd_df[target_var]).year
    pd_df['{0}_month'.format(prefix)] = pd.DatetimeIndex(pd_df[target_var]).month
    pd_df['{0}_day'.format(prefix)]   = pd.DatetimeIndex(pd_df[target_var]).day
    pd_df['{0}_weekday'.format(prefix)]   = pd.DatetimeIndex(pd_df[target_var]).weekday
    pd_df['{0}_hour'.format(prefix)]  = pd.DatetimeIndex(pd_df[target_var]).hour
    pd_df.drop(target_var, inplace=True, axis=1)
    return pd_df

hpg_reserve = extract_dates(pd_df = hpg_reserve, target_var = 'visit_datetime', prefix='target')
hpg_reserve = extract_dates(pd_df = hpg_reserve, target_var = 'reserve_datetime')
hpg_reserve.head()

air_reserve = extract_dates(pd_df = air_reserve, target_var = 'visit_datetime', prefix='target')
air_reserve = extract_dates(pd_df = air_reserve, target_var = 'reserve_datetime')
air_reserve.pivot_table(columns='reserve_datetime_weekday')

date_info = extract_dates(pd_df = date_info, target_var = 'calendar_date', format_str="%Y-%m-%d", prefix='date')
date_info.drop('date_hour', inplace=True, axis=1)
date_info.pivot_table(columns='day_of_week')

#%%
# Extract dates and location from submission table
# First, split up id column

x = sample_submission.join(sample_submission['id'].str.split('_', 1, expand=True).rename(columns={0:'id1', 1:'id2'}))
x['id2'], x['date'] = x['id2'].str.split('_', 1).str
x['air_store_id'] = x[['id1', 'id2']].apply(lambda x: '_'.join(x), axis=1)
x.rename(columns={'id1':'store_type'}, inplace=True)
x.drop('id2', inplace=True, axis=1)
x = extract_dates(pd_df = x, target_var = 'date', format_str="%Y-%m-%d", prefix='target')
x.drop('target_hour', inplace=True, axis=1)
x['test'] = 1
x.head()
sample_submission = x
sample_submission.head()

#%%

sample_submission.pivot_table(columns='store_type')

#%%
# Extract date info from air_visit_data

air_visit_data = extract_dates(pd_df = air_visit_data, target_var = 'visit_date', 
                               format_str="%Y-%m-%d", prefix='target')
air_visit_data.drop('target_hour', axis=1, inplace=True)
air_visit_data['id'] = 0
air_visit_data['test'] = 0
air_visit_data['store_type'] = 'air'
air_visit_data.head()

#%%
# Constructing main table
# Concatenate edited submission table with air_visit_data

main_tbl = pd.concat([air_visit_data, sample_submission])
main_tbl.head()

#%%

main_tbl.pivot_table(columns='test')

#%%
# Prepare baseline submission
# Ie. average number of visitors for each restaurant.

def prepare_submission(main_tbl, fpath=None):
    sub_tbl = main_tbl.loc[main_tbl['test'] == 1][['id', 'visitors']]
    if fpath:
        sub_tbl.to_csv(fpath, index=False)
    return sub_tbl

#%%
# Get mean visitor numbers for train data
visitors_mean = main_tbl.loc[main_tbl['test'] == 0].visitors.mean()
baseline_sub = main_tbl.loc[main_tbl['test'] == 1][['id', 'visitors']]
baseline_sub.visitors = visitors_mean
baseline_sub.to_csv('submissions/baseline.csv', index=False)

#%%
# Merge with additional infos

#%%
# ### Merge with store info

air_store_info.head()

main_tbl_merge = pd.merge(main_tbl, air_store_info, on='air_store_id')
main_tbl_merge.head()

main_tbl_merge.describe()

#%%
# Merge with holiday info.

date_info_merge = date_info
date_info_merge['date_id'] = date_info_merge[['date_year', 'date_month', 'date_day']].astype(str).apply(lambda x: '_'.join((x)), axis=1)
date_info_merge = date_info_merge[['holiday_flg', 'date_id']]
date_info_merge.head()

#%%

main_tbl_merge2 = main_tbl_merge
main_tbl_merge2['date_id'] = main_tbl_merge2[['target_year', 'target_month', 'target_day']].astype(str).apply(lambda x: '_'.join((x)), axis=1)

main_tbl_merge2.head()

#%%

main_tbl_merge = pd.merge(left=main_tbl_merge2, right=date_info_merge, on='date_id')
main_tbl_merge.describe()

#%%
# Save current stage to disk

main_tbl_merge.to_csv('output/main_tbl.csv')
model_data = pd.read_csv('output/main_tbl.csv')
target_cols = ['target_day', 'target_month', 'target_weekday', 'target_year', 'latitude', 'longitude', 'holiday_flg',
              'test','visitors']
X = model_data[target_cols]

#%% Plotting 

import matplotlib.pyplot as plt
import numpy as np
def plot_corr(df, size=16):
    font={'size':16}
    plt.rc('font',**font)
    corr=df.corr()
    fig, ax = plt.subplots(figsize=(size,size))
    cax=ax.matshow(corr, interpolation='nearest')
    fig.colorbar(cax)
    ax.matshow(corr)
    plt.show()
    print(corr.columns)

plot_corr(X[X['test']==0])
# It looks like there is a strong correlation between latitude and longitude, can we combine them into a single feature?

#%%
# ## First simple modelling attempt

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#Suppress all futurewarnings in console
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


target_cols_fit = [col for col in X.columns if not col in ['test','visitors']]
Xsub=X[X['test']==1]
X=X[X['test']==0]

Xsub = Xsub.drop(['test'],axis=1)
X = X.drop(['test'],axis=1)

mod = SGDRegressor()

pipe = Pipeline([('scal',StandardScaler()),
                 ('clf',mod)])

X_train, X_test, y_train, y_test = train_test_split(X[target_cols_fit], X['visitors'],test_size=0.15, random_state=42)

parameters = {'clf__alpha': np.logspace(-4,1,6)}
pipe = GridSearchCV(pipe, parameters)

pipe.fit(X_train,y_train)
print(pipe.best_params_)

#%%
# Create a table of real vs predicted
testIndices=X_test.index.values
prediction = pd.Series(pipe.predict(X_test), name='Prediction', index=testIndices)
prediction2 = pd.read_csv('submissions/xg_boost1.csv')
y_test.name='Real'
resultsvsPredictions = pd.concat([prediction, y_test], axis=1)

scores = cross_val_score(pipe,X_test,y_test, scoring='neg_mean_squared_error')
scores_base=cross_val_score(pipe, X_test,pd.Series(np.ones(len(y_test))*y_test.mean()))

print("Trained mean squared error is {:.1f} and untrained is {:.1f}"\
          .format(scores.mean(),scores_base.mean()))

#%% Do predictions for submission

Xsub['visitors']=pipe.predict(Xsub[target_cols_fit])
SGD_sub=baseline_sub
SGD_sub.visitors = pipe.predict(Xsub[target_cols_fit])
SGD_sub.to_csv('submissions/SGD.csv', index=False)
Xsub.pivot_table(columns='holiday_flg')
