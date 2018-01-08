
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
# Remember any features extracted from training tables need to be extractable
# from submission table.
#
# Remember that we are predicting visits, not reservations.
#
# Check if we have reservation data for some or all submission dates.
#  - We have reservation data for test and training data.
#
# Check if the restaurants in the submission list are all included in the
# training data.
#
# Latitude and logitude are highly correlated: can we combine to a single
# feature?
#
# ## For the time-series modelling, how should we organise the data?
# * Average over all restaurants
# * Treat every restaurant separately
# * Combine restaurants by genre
# * Combine restaurants by geographical location (create clusters maybe)
# * fbprophet allows a holiday flag to be added.

# ## Loading modules

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
import logging
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product
# Suppress all futurewarnings in console
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('fbprophet.forecaster').propagate = False


# ## Loading raw data
air_reserve = pd.read_csv('input/air_reserve.csv')
air_store_info = pd.read_csv('input/air_store_info.csv')
air_visit_data = pd.read_csv('input/air_visit_data.csv')
hpg_reserve = pd.read_csv('input/hpg_reserve.csv')
hpg_store_info = pd.read_csv('input/hpg_store_info.csv')
date_info = pd.read_csv('input/date_info.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')
store_id_relation = pd.read_csv('input/store_id_relation.csv')

# %% 
# Prepare data for submission


def prepareFBProphetSubmission(sample_submission):
    """ Prepare a submission table for use in fbprophet predictions """
    subTable = sample_submission.join(sample_submission['id'].str.
                                      split('_', 1, expand=True).
                                      rename(columns={0: 'id1', 1: 'id2'}))
    subTable['id2'], subTable['ds'] = subTable['id2'].str.split('_', 1).str
    subTable['air_store_id'] = subTable[['id1', 'id2']]\
                                .apply(lambda x: '_'.join(x), axis=1)
    subTable.drop(['id', 'id1', 'id2'], inplace=True, axis=1)

    return subTable


subTable = prepareFBProphetSubmission(sample_submission)


# %%
# Fbprophet - Average over all restaurants

# def timeSeriesPreprocessing(air_visit_data):
#    """ Preprocessing of data for fbprophet time-series modelling """

prophetData = air_visit_data[['visit_date', 'visitors']]
allDays = pd.date_range(prophetData.visit_date.min(),
                        prophetData.visit_date.max())
holidays = date_info.loc[date_info['holiday_flg'] == 1]
holidays = holidays[['calendar_date']]
holidays.columns = ['ds']
holidays['holiday'] = 'Holiday'

prophetData.columns = ['ds', 'y']
newProphet = prophetData.groupby(['ds'], axis=0).mean().reset_index()
prophetTrain, prophetTest = train_test_split(newProphet, test_size=0.25,
                                             random_state=42)
m = Prophet(holidays=holidays)
m.fit(prophetTrain)
testForecast = m.predict(prophetTest[['ds']])

errorBulk = mean_squared_error(prophetTest['y'], testForecast['yhat'])
print("Error when trained in bulk is {:.1f}".format(errorBulk))
# Would mean absolute error be a better metric?

uniqueForecastDates = pd.DataFrame(
        subTable.ds.unique().astype(str), columns=['ds'])
subForecast = m.predict(uniqueForecastDates)
subForecast['ds'] = subForecast['ds'].astype(str)

avgSubTable = subTable.merge(subForecast[['ds', 'yhat']], on='ds')
avgSubTable['air_store_id'] = avgSubTable[['air_store_id', 'ds']].\
                                apply(lambda x: '_'.join(x), axis=1)
avgSubTable.drop(['visitors', 'ds'], inplace=True, axis=1)
avgSubTable.columns = ['id', 'visitors']
avgSubTable.to_csv('submissions/avgFbProphet.csv', index=False)


# %%
# FBProphet - Predict each name separately 

prophetData = air_visit_data

uniqueNames = prophetData.air_store_id.unique()
allNameDates = pd.DataFrame(list(product(uniqueNames,uniqueForecastDates['ds'])),
                             columns=['genres', 'ds'])
# forecastDates = 
totalPred = pd.DataFrame()
loopNo=0

for storeName in uniqueNames:
    m = Prophet(holidays=holidays)
    thisProphet = prophetData[prophetData['air_store_id']==storeName]
    thisProphet.drop(['air_store_id'], inplace=True, axis=1)
    thisProphet['visit_date'] = pd.to_datetime(thisProphet['visit_date'])
    thisProphet = thisProphet.set_index('visit_date')
    thisProphet = thisProphet.reindex(allDays).fillna(
            thisProphet.visitors.mean()).reset_index()
    thisProphet.columns = ['ds','y']
    m.fit(thisProphet)
    thisSubPred = m.predict(uniqueForecastDates)
    thisSubPred['ds'] = thisSubPred['ds'].astype(str)
    totalPred[storeName] = thisSubPred[['yhat']]
    print(loopNo)
    loopNo = loopNo + 1
    
totalPred = pd.melt(totalPred)
totalPred = totalPred.rename(columns={'variable': 'air_store_id', 'value': 'y'})
totalPred['ds'] = allNameDates['ds']
totalPred.y.loc[totalPred['y'] < 0] = 0
storeSubTable = subTable.merge(totalPred, on=['ds', 'air_store_id'])
storeSubTable['air_store_id'] = storeSubTable[['air_store_id', 'ds']].\
                                apply(lambda x: '_'.join(x), axis=1)
storeSubTable.drop(['visitors','ds'], axis=1, inplace=True)
storeSubTable.columns = ['id', 'visitors']
storeSubTable.to_csv('submissions/storeFbProphet.csv', index=False)

# %%
# FBProphet - Average over genres
prophetData = pd.concat([air_visit_data[['visit_date', 'visitors']],
                         air_store_info[['air_genre_name']]], axis=1)
newProphet = prophetData.groupby(['air_genre_name', 'visit_date'], axis=0)\
                                                    .mean().reset_index()

uniqueGenres = newProphet.air_genre_name.unique().astype(str)

errorSeparate = []
noTrainingPoints = []
totalPred = pd.DataFrame()
# Predict the average if there are too few training points. 
minNoTrainingPoints = 20 
for genre in uniqueGenres:
    m = Prophet(holidays=holidays)
    thisProphet = newProphet.loc[newProphet['air_genre_name'] == genre]
    thisProphet = thisProphet.drop(['air_genre_name'], axis=1)
    thisProphet['visit_date'] = pd.to_datetime(thisProphet['visit_date'])
    thisProphet = thisProphet.set_index('visit_date')
    noTrainingPoints.append(len(thisProphet))
    if len(thisProphet) < minNoTrainingPoints:
        thisProphet = thisProphet.reindex(allDays).fillna(
                thisProphet.visitors.mean()).reset_index()
    else:
        thisProphet = thisProphet.reindex(allDays).reset_index()
    thisProphet.columns = ['ds', 'y']

    m.fit(thisProphet)
    thisSubPred = m.predict(uniqueForecastDates)
    # m.plot(thisSubPred)
    thisSubPred['ds'] = thisSubPred['ds'].astype(str)
    totalPred[genre] = thisSubPred[['yhat']]

totalPred = pd.melt(totalPred)
totalPred = totalPred.rename(columns={'variable': 'genres', 'value': 'y'})
totalPred['ds'] = allGenreDates['ds']
totalPred.y.loc[totalPred['y'] < 0] = 0
genreSubTable = subTable.merge(air_store_info[['air_store_id','air_genre_name']],
                               on='air_store_id')
genreSubTable = genreSubTable.merge(totalPred, left_on=['air_genre_name','ds'],
                                    right_on=['genres','ds'])
genreSubTable['air_store_id'] = genreSubTable[['air_store_id', 'ds']].\
                                apply(lambda x: '_'.join(x), axis=1)
genreSubTable.drop(['visitors','ds','air_genre_name','genres'], 
                   axis=1, inplace=True)
genreSubTable.columns = ['id', 'visitors']
genreSubTable.to_csv('submissions/genreFbProphet.csv', index=False)

# Note: Some of these have so little data that predictions are completely off.
# This is especially significant if we use area rather than genre. Can we
# cluster the restaurants somehow and train each cluster separately?

# It's not fair to fill NaNs and then split - the predictions will then
# be great when there are very few data points. Not really any way to get
# around this however, because some genres just don't have enough data points.
# *Just make a prediction based on all of the data and submit data to kaggle
#  to find out the score

# Is the data definitely being taken correctly here?

# %%
# Cluster data geographically and then train - distances are close so
# use k-means rather than DBSCAN.

prophetData = pd.merge(air_visit_data,
                       air_store_info[['air_store_id',
                                       'latitude', 'longitude']],
                       on='air_store_id')
# plt.scatter(air_store_info['latitude'],air_store_info['longitude'])

kmeans = KMeans(n_clusters=4, random_state=42).\
                fit(air_store_info[['latitude', 'longitude']])
prophetData['clusterNo'] = kmeans.predict(prophetData[['latitude',
                                                       'longitude']])
prophetData = prophetData.drop(['latitude', 'longitude'], axis=1)
# prophetData.hist(column='clusterNo')

newProphet = prophetData.groupby(['clusterNo', 'visit_date'], axis=0)\
                                                    .mean().reset_index()

uniqueClusters = newProphet.clusterNo.unique()
errorClusters = []

for cluster in uniqueClusters:
    m = Prophet(holidays=holidays)
    thisProphet = newProphet.loc[newProphet['clusterNo'] == cluster]
    thisProphet = thisProphet.drop(['clusterNo'], axis=1)
    thisProphet['visit_date'] = pd.to_datetime(thisProphet['visit_date'])
    thisProphet = thisProphet.set_index('visit_date')
    thisProphet = thisProphet.reindex(allDays).fillna(thisProphet.visitors.
                                                      mean()).reset_index()
    thisProphet.columns = ['ds', 'y']
    prophetTrain, prophetTest = train_test_split(thisProphet, test_size=0.15,
                                                 random_state=42)

    m.fit(prophetTrain)
    forecast = m.predict(pd.DataFrame(prophetTest['ds']))
    errorClusters.append(mean_squared_error(prophetTest['y'],
                                            forecast['yhat']))
    # m.plot(forecast)
    # m.plot_components(forecast);

print("Error when trained using separate clusters is {:.1f}"
      .format(np.mean(errorClusters)))

# %%
# Extract info from dates
# First, define a little function to extract year, month, day, hour


def extract_dates(pd_df, target_var, format_str="%Y-%m-%d %H:%M:%S",
                  prefix=None):
    """Extract the year, month, day, weekday and hour from the data for use as
    features"""
    if not prefix:
        prefix = target_var
    pd_df[target_var] = pd.to_datetime(pd_df[target_var], format=format_str)
    pd_df['{0}_year'.format(prefix)] = pd.DatetimeIndex(
            pd_df[target_var]).year
    pd_df['{0}_month'.format(prefix)] = pd.DatetimeIndex(
            pd_df[target_var]).month
    pd_df['{0}_day'.format(prefix)] = pd.DatetimeIndex(
            pd_df[target_var]).day
    pd_df['{0}_weekday'.format(prefix)] = pd.DatetimeIndex(
            pd_df[target_var]).weekday
    pd_df['{0}_hour'.format(prefix)] = pd.DatetimeIndex(
            pd_df[target_var]).hour
    pd_df.drop(target_var, inplace=True, axis=1)
    return pd_df


# %%


def SGDPreprocessing(air_reserve, air_store_info, air_visit_data, hpg_reserve,
                     hpg_store_info, date_info, sample_submission,
                     store_id_relation):
    """ Preprocessing of data for stochastic gradient descent modelling """

    hpg_reserve = extract_dates(pd_df=hpg_reserve,
                                target_var='visit_datetime', prefix='target')
    hpg_reserve = extract_dates(pd_df=hpg_reserve,
                                target_var='reserve_datetime')
    hpg_reserve.head()
    air_reserve = extract_dates(pd_df=air_reserve,
                                target_var='visit_datetime', prefix='target')
    air_reserve = extract_dates(pd_df=air_reserve,
                                target_var='reserve_datetime')
    air_reserve.pivot_table(columns='reserve_datetime_weekday')

    date_info = extract_dates(pd_df=date_info, target_var='calendar_date',
                              format_str="%Y-%m-%d", prefix='date')
    date_info.drop('date_hour', inplace=True, axis=1)
    date_info.pivot_table(columns='day_of_week')

    air_visit_data = extract_dates(pd_df=air_visit_data,
                                   target_var='visit_date',
                                   format_str="%Y-%m-%d", prefix='target')
    air_visit_data.drop('target_hour', axis=1, inplace=True)
    air_visit_data['id'] = 0
    air_visit_data['test'] = 0
    air_visit_data['store_type'] = 'air'
    air_visit_data.head()

    # %%
    # Extract dates and location from submission table
    # First, split up id column

    x = sample_submission.join(sample_submission['id'].str.split('_', 1,
                               expand=True).rename(
                                       columns={0: 'id1', 1: 'id2'}))
    x['id2'], x['date'] = x['id2'].str.split('_', 1).str
    x['air_store_id'] = x[['id1', 'id2']].apply(lambda x: '_'.join(x), axis=1)
    x.rename(columns={'id1': 'store_type'}, inplace=True)
    x.drop('id2', inplace=True, axis=1)
    x = extract_dates(pd_df=x, target_var='date', format_str="%Y-%m-%d",
                      prefix='target')
    x.drop('target_hour', inplace=True, axis=1)
    x['test'] = 1
    x.head()
    sample_submission = x
    sample_submission.head()

    # %%

    sample_submission.pivot_table(columns='store_type')

    # %%
    # Constructing main table
    # Concatenate edited submission table with air_visit_data

    main_tbl = pd.concat([air_visit_data, sample_submission])
    main_tbl.head()

    # %%

    main_tbl.pivot_table(columns='test')

    # %%
    # Get mean visitor numbers for train data
    visitors_mean = main_tbl.loc[main_tbl['test'] == 0].visitors.mean()
    baseline_sub = main_tbl.loc[main_tbl['test'] == 1][['id', 'visitors']]
    baseline_sub.visitors = visitors_mean
    baseline_sub.to_csv('submissions/baseline.csv', index=False)

    # %%
    # Merge with additional infos

    # %%
    # Merge with store info

    air_store_info.head()

    main_tbl_merge = pd.merge(main_tbl, air_store_info, on='air_store_id')
    main_tbl_merge.head()

    main_tbl_merge.describe()

    # %%
    # Merge with holiday info.

    date_info_merge = date_info
    date_info_merge['date_id'] = date_info_merge[['date_year',
                                                  'date_month', 'date_day']].\
                                                   astype(str).\
                                                   apply(lambda x: '_'.
                                                         join((x)), axis=1)
    date_info_merge = date_info_merge[['holiday_flg', 'date_id']]
    date_info_merge.head()

    # %%

    main_tbl_merge2 = main_tbl_merge
    main_tbl_merge2['date_id'] = main_tbl_merge2[['target_year',
                                                  'target_month', 'target_day']].\
                                                   astype(str).\
                                                   apply(lambda x: '_'.
                                                         join((x)), axis=1)

    main_tbl_merge2.head()

    # %%

    main_tbl_merge = pd.merge(left=main_tbl_merge2, right=date_info_merge,
                              on='date_id')
    main_tbl_merge.describe()

    # %%
    # Save current stage to disk

    main_tbl_merge.to_csv('output/main_tbl.csv')
    return 0

# %% Plotting


def plot_corr(size=8):
    """ Plot the correlation between features"""
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'test', 'visitors']
    X = model_data[target_cols]

    font = {'size': 16}
    plt.rc('font', **font)
    corr = X[X['test'] == 0].corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, interpolation='nearest')
    fig.colorbar(cax)
    ax.matshow(corr)
    plt.show()
    print(corr.columns)

# plot_corr()

# It looks like there is a strong correlation between latitude and longitude,
# can we combine them into a single feature?

# %%
# ## First simple modelling attempt


def SGDFit():
    """ Fit with a stochastic gradient descent classifier """
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'test', 'visitors']
    X = model_data[target_cols]
    target_cols_fit = [col for col in X.columns if col not in ['test',
                                                               'visitors']]
    Xsub = X[X['test'] == 1]
    X = X[X['test'] == 0]

    Xsub = Xsub.drop(['test'], axis=1)
    X = X.drop(['test'], axis=1)

    mod = SGDRegressor()

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', mod)])

    X_train, X_test, y_train, y_test = train_test_split(X[target_cols_fit],
                                                        X['visitors'],
                                                        test_size=0.15,
                                                        random_state=42)

    parameters = {'clf__alpha': np.logspace(-4, 1, 6)}
    pipe = GridSearchCV(pipe, parameters)

    pipe.fit(X_train, y_train)
    print(pipe.best_params_)

    # Create a table of real vs predicted
    testIndices = X_test.index.values
    prediction = pd.Series(pipe.predict(X_test), name='Prediction',
                           index=testIndices)
    y_test.name = 'Real'
    # resultsvsPredictions = pd.concat([prediction, y_test], axis=1)

    scores = cross_val_score(pipe, X_test, y_test,
                             scoring='neg_mean_squared_error')
    # scores = mean_squared_error(y_test,prediction)
    scores_base = cross_val_score(pipe, X_test,
                                  pd.Series(
                                          np.ones(len(y_test))*y_test.mean()))

    print("Trained mean squared error is {:.1f} and untrained is {:.1f}"
          .format(np.abs(scores.mean()), np.abs(scores_base.mean())))

    # Do predictions for submission

    Xsub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    SGD_sub = pd.read_csv('submissions/baseline.csv')
    SGD_sub.visitors = pipe.predict(Xsub[target_cols_fit])
    SGD_sub.to_csv('submissions/SGD.csv', index=False)
    Xsub.pivot_table(columns='holiday_flg')

# SGDPreprocessing(air_reserve, air_store_info, air_visit_data,
#                 hpg_reserve, hpg_store_info, date_info,
#                 sample_submission, store_id_relation)
    
# SGDFit()
