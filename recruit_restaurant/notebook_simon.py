
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
# TODO: Add number of observations for each restaurant and day of week as a 
# feature. Add genre and area names as feature. 

# ## Loading modules

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import logging
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import product
import xgboost as xgb
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


def prepareFBProphetSubmission(sample_submission):
    """ Prepare a submission table for use in fbprophet predictions """
    subTable = sample_submission.join(sample_submission['id'].str.
                                      split('_', 1, expand=True).
                                      rename(columns={0: 'id1', 1: 'id2'}))
    subTable['id2'], subTable['ds'] = subTable['id2'].str.split('_', 1).str
    subTable['air_store_id'] = subTable[['id1', 'id2']]\
        .apply(lambda x: '_'.join(x), axis=1)
    subTable.drop(['id', 'id1', 'id2', 'visitors'], inplace=True, axis=1)

    return subTable


def dates(date_info, air_visit_data, subTable):
    """ Return a dataframe of holidays """

    holidays = date_info.loc[date_info['holiday_flg'] == 1]
    holidays = holidays[['calendar_date']]
    holidays.columns = ['ds']
    holidays['holiday'] = 'Holiday'

    trainDays = pd.DataFrame(air_visit_data.visit_date.unique(),
                             columns=['ds'])
    trainDays = trainDays.sort_values('ds')
    forecastDays = pd.DataFrame(subTable.ds.unique(), columns=['ds'])

    return holidays, trainDays, forecastDays


def avgFBProphet(air_visit_data, subTable, forecastDays, holidays):
    """ FBProphet run using a sum off all visits on each date """

    prophetData = air_visit_data[['visit_date', 'visitors']]
    prophetData.columns = ['ds', 'y']
    newProphet = prophetData.groupby(['ds'], axis=0).mean().reset_index()
    m = Prophet(holidays=holidays)
    m.fit(newProphet)
    subForecast = m.predict(forecastDays)
    subForecast['ds'] = subForecast['ds'].astype(str)

    avgSubTable = subTable.merge(subForecast[['ds', 'yhat']], on='ds')
    avgSubTable['air_store_id'] = avgSubTable[['air_store_id', 'ds']].\
        apply(lambda x: '_'.join(x), axis=1)
    avgSubTable.drop(['ds'], inplace=True, axis=1)
    avgSubTable.columns = ['id', 'visitors']
    avgSubTable.to_csv('submissions/avgFbProphet.csv', index=False)

    return avgSubTable


def nameFBProphet(air_visit_data, forecastDays, holidays, trainDays, subTable):
    """ FBProphet run for each restaurant separately """
    prophetData = air_visit_data
    uniqueNames = prophetData.air_store_id.unique()
    allNameDates = pd.DataFrame(list(product(uniqueNames, forecastDays['ds'])),
                                columns=['genres', 'ds'])

    totalPred = pd.DataFrame()
    loopNo = 0

    for storeName in uniqueNames:
        m = Prophet(holidays=holidays)
        thisProphet = prophetData[prophetData['air_store_id'] == storeName]
        thisProphet.drop(['air_store_id'], inplace=True, axis=1)
        thisProphet['visit_date'] = pd.to_datetime(thisProphet['visit_date'])
        thisProphet = thisProphet.set_index('visit_date')
        thisProphet = thisProphet.reindex(trainDays['ds'])
        thisProphet = thisProphet.fillna(0).reset_index()
        thisProphet.columns = ['ds', 'y']
        m.fit(thisProphet)
        thisSubPred = m.predict(forecastDays)
        m.plot(thisSubPred)
        thisSubPred['ds'] = thisSubPred['ds'].astype(str)
        totalPred[storeName] = thisSubPred[['yhat']]
        print(str(loopNo)+" ", end='')
        loopNo = loopNo + 1

    totalPred = pd.melt(totalPred)
    totalPred = totalPred.rename(columns={'variable': 'air_store_id',
                                          'value': 'y'})
    totalPred['ds'] = allNameDates['ds']
    totalPred.y.loc[totalPred['y'] < 0] = 0
    storeSubTable = subTable.merge(totalPred, on=['ds', 'air_store_id'])
    storeSubTable['air_store_id'] = storeSubTable[['air_store_id', 'ds']].\
        apply(lambda x: '_'.join(x), axis=1)
    storeSubTable.drop(['ds'], axis=1, inplace=True)
    storeSubTable.columns = ['id', 'visitors']
    storeSubTable.to_csv('submissions/storeFbProphet.csv', index=False)

    return storeSubTable


def genreFBProphet(air_visit_data, air_store_info, holidays, trainDays,
                   forecastDays, subTable):
    """ FBProphet run for each genre separately """

    prophetData = pd.concat([air_visit_data[['visit_date', 'visitors']],
                             air_store_info[['air_genre_name']]], axis=1)
    newProphet = prophetData.groupby(['air_genre_name', 'visit_date'], axis=0)\
        .mean().reset_index()

    uniqueGenres = newProphet.air_genre_name.unique().astype(str)
    allGenreDates = pd.DataFrame(list(product(uniqueGenres,forecastDays['ds'])),
                                 columns=['genres', 'ds'])

    totalPred = pd.DataFrame()
    # Predict from 0 if there are too few training points.
    minNoTrainingPoints = 20
    for genre in uniqueGenres:
        m = Prophet(holidays=holidays)
        thisProphet = newProphet.loc[newProphet['air_genre_name'] == genre]
        thisProphet = thisProphet.drop(['air_genre_name'], axis=1)
        thisProphet = thisProphet.set_index('visit_date')
        if len(thisProphet) < minNoTrainingPoints:
            thisProphet = thisProphet.reindex(trainDays['ds'])
            thisProphet = thisProphet.fillna(0).reset_index()
        else:
            thisProphet = thisProphet.reindex(trainDays['ds']).reset_index()
        thisProphet.columns = ['ds', 'y']

        m.fit(thisProphet)
        thisSubPred = m.predict(forecastDays)
        thisSubPred['ds'] = thisSubPred['ds'].astype(str)
        totalPred[genre] = thisSubPred[['yhat']]

    totalPred = pd.melt(totalPred)
    totalPred = totalPred.rename(columns={'variable': 'genres', 'value': 'y'})
    totalPred['ds'] = allGenreDates['ds']
    totalPred.y.loc[totalPred['y'] < 0] = 0
    genreSubTable = subTable.merge(air_store_info[['air_store_id',
                                                   'air_genre_name']],
                                   on='air_store_id')
    genreSubTable = genreSubTable.merge(totalPred,
                                        left_on=['air_genre_name', 'ds'],
                                        right_on=['genres', 'ds'])
    genreSubTable['air_store_id'] = genreSubTable[['air_store_id', 'ds']].\
        apply(lambda x: '_'.join(x), axis=1)
    genreSubTable.drop(['ds', 'air_genre_name', 'genres'],
                       axis=1, inplace=True)
    genreSubTable.columns = ['id', 'visitors']
    genreSubTable.to_csv('submissions/genreFbProphet.csv', index=False)

    return genreSubTable

    # Note: Some of these have so little data that predictions are completely
    # off. This is especially significant if we use area rather than genre.
    # It's not fair to fill NaNs and then split. Not really any way to get
    # around this however, because some genres just don't have enough data
    # points.


def clusterFBProphet(air_visit_data, air_store_info, subTable, holidays,
                     trainDays, forecastDays):
    """ Cluster geographically by latitude and longitude then run FBProphet
    for each cluster separately. Distances are close so k-means is used rather
    than something like DBSCAN"""

    prophetData = pd.merge(air_visit_data, air_store_info[['air_store_id',
                                                           'latitude',
                                                           'longitude']],
                           on='air_store_id')
    # plt.scatter(air_store_info['latitude'],air_store_info['longitude'])
    nClusters = 4
    clusterList = list(range(nClusters))
    kmeans = KMeans(n_clusters=nClusters).fit(air_store_info[['latitude',
                                                              'longitude']])
    allClusterDates = pd.DataFrame(list(product(clusterList,
                                                forecastDays['ds'])),
                                   columns=['genres', 'ds'])
    prophetData['clusterNo'] = kmeans.predict(prophetData[['latitude',
                                                           'longitude']])
    subTable = subTable.merge(air_store_info[['air_store_id', 'latitude',
                                              'longitude']],
                              on='air_store_id')
    subTable['clusterNo'] = kmeans.predict(subTable[['latitude', 'longitude']])
    subTable.drop(['latitude', 'longitude'], axis=1, inplace=True)
    prophetData = prophetData.drop(['latitude', 'longitude'], axis=1)
    # prophetData.hist(column='clusterNo')

    newProphet = prophetData.groupby(['clusterNo', 'visit_date'],
                                     axis=0).mean().reset_index()
    totalPred = pd.DataFrame()
    for cluster in clusterList:
        m = Prophet(holidays=holidays)
        thisProphet = newProphet.loc[newProphet['clusterNo'] == cluster]
        thisProphet = thisProphet.drop(['clusterNo'], axis=1)
        thisProphet['visit_date'] = pd.to_datetime(thisProphet['visit_date'])
        thisProphet = thisProphet.set_index('visit_date')
        thisProphet = thisProphet.reindex(trainDays['ds'])
        thisProphet = thisProphet.fillna(0).reset_index()
        thisProphet.columns = ['ds', 'y']

        m.fit(thisProphet)
        thisSubPred = m.predict(forecastDays)
        thisSubPred['ds'] = thisSubPred['ds'].astype(str)
        totalPred[cluster] = thisSubPred[['yhat']]

    totalPred = pd.melt(totalPred)
    totalPred = totalPred.rename(columns={'variable': 'clusterNo',
                                          'value': 'y'})
    totalPred['ds'] = allClusterDates['ds']
    totalPred.y.loc[totalPred['y'] < 0] = 0

    clusterSubTable = subTable.merge(totalPred, on=['clusterNo', 'ds'])
    clusterSubTable['air_store_id'] = clusterSubTable[['air_store_id', 'ds']].\
        apply(lambda x: '_'.join(x), axis=1)

    clusterSubTable.drop(['ds', 'clusterNo'],
                         axis=1, inplace=True)
    clusterSubTable.columns = ['id', 'visitors']
    clusterSubTable.to_csv('submissions/clusterFbProphet.csv', index=False)

    return clusterSubTable


# subTable = prepareFBProphetSubmission(sample_submission)
# holidays, trainDays, forecastDays = dates(date_info, air_visit_data, subTable)
# avgSubTable = avgFBProphet(air_visit_data, subTable, forecastDays, holidays)
# storeSubTable = nameFBProphet(air_visit_data, forecastDays, holidays, trainDays)
# genreSubTable = genreFBProphet(air_visit_data, air_store_info, holidays,
#                                trainDays, forecastDays)
# clusterSubTable = clusterFBProphet(air_visit_data, air_store_info, subTable,
#                                    forecastDays)


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


def SGDPreprocessing(air_reserve, air_store_info, air_visit_data, hpg_reserve,
                     hpg_store_info, date_info, sample_submission,
                     store_id_relation):
    """ Preprocessing of data for stochastic gradient descent modelling """
    # Join real data with data to predict into one dataframe, with a 'test'
    # label to define if training or test data.
    air_visit_data['test'] = 0
    future = sample_submission.join(sample_submission['id'].str.split('_', 1,
                                    expand=True).rename(
                                    columns={0: 'id1', 1: 'id2'}))
    future['id2'], future['visit_date'] = future['id2'].str.split('_', 1).str
    future['air_store_id'] = future[['id1', 'id2']].apply(
                                                    lambda x: '_'.join(x),
                                                    axis=1)
    future.drop(['id', 'id1', 'id2'], inplace=True, axis=1)
    future['test'] = 1
    future['visitors'] = 0
    mainTable = air_visit_data.append(future, ignore_index=True)

    # Add air reservation data to table, filling NaNs with zero
    air_reserve['visit_date'] = air_reserve['visit_datetime'].apply(
                                                            lambda x: x[:10])
    air_reserve.drop(['reserve_datetime', 'visit_datetime'], axis=1,
                     inplace=True)
    air_reserve = air_reserve.groupby(['air_store_id', 'visit_date'], axis=0)\
                                       .sum().reset_index()
    mainTable = mainTable.merge(air_reserve, how='left',
                                on=['air_store_id', 'visit_date'])
    mainTable['reserve_visitors'] = mainTable['reserve_visitors'].fillna(0)

    # Add hpg reservation data to table, filling NaNs with zero

    hpg_reserve = hpg_reserve.merge(store_id_relation, how='left',
                                    on='hpg_store_id')
    hpg_reserve = hpg_reserve.dropna()
    hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].apply(
                                                            lambda x: x[:10])
    hpg_reserve.drop(['reserve_datetime', 'visit_datetime', 'hpg_store_id'],
                     axis=1, inplace=True)
    hpg_reserve = hpg_reserve.groupby(['air_store_id', 'visit_date'], axis=0)\
                                       .sum().reset_index()
    hpg_reserve.columns = ['air_store_id', 'visit_date', 'hpg_res_visitors']
    mainTable = mainTable.merge(hpg_reserve, how='left',
                                on=['air_store_id', 'visit_date'])
    mainTable['hpg_res_visitors'] = mainTable['hpg_res_visitors'].fillna(0)
    mainTable['reserve_visitors'] = mainTable['reserve_visitors'] +\
                                    mainTable['hpg_res_visitors']
    mainTable.drop(['hpg_res_visitors'], axis=1, inplace=True)

    # Get mean visitor numbers for train data and prepare baseline submission
    visitorsMean = mainTable.loc[mainTable['test'] == 0].visitors.mean()
    baselineSub = mainTable
    baselineSub['id'] = baselineSub['air_store_id'].map(str) +\
                                                         "_" +\
                                                    baselineSub['visit_date']
    baselineSub = baselineSub.loc[mainTable['test'] == 1][['id', 'visitors']]
    baselineSub.visitors = visitorsMean
    baselineSub.to_csv('submissions/baseline.csv', index=False)

    # Add features to main table
    mainTable = pd.merge(mainTable, air_store_info, on='air_store_id')
    date_info.columns = ['visit_date', 'day', 'holiday_flg']
    mainTable = pd.merge(mainTable, date_info[['visit_date', 'holiday_flg']],
                         on='visit_date')
    mainTable = extract_dates(pd_df=mainTable,
                              target_var='visit_date',
                              format_str="%Y-%m-%d", prefix='target')
    mainTable.drop('target_hour', inplace=True, axis=1)

    # Add min, max and median visitors as a feature

    temp = mainTable.groupby(['air_store_id', 'target_weekday'],
                             as_index=False)['visitors'].min().rename(
                                     columns={'visitors': 'min_visitors'})
    mainTable = pd.merge(mainTable, temp, how='left', 
                         on=['air_store_id', 'target_weekday'])
    
    temp = mainTable.groupby(['air_store_id', 'target_weekday'],
                             as_index=False)['visitors'].max().rename(
                                     columns={'visitors': 'max_visitors'})
    mainTable = pd.merge(mainTable, temp, how='left', 
                         on=['air_store_id', 'target_weekday'])
    
    temp = mainTable.groupby(['air_store_id', 'target_weekday'],
                             as_index=False)['visitors'].median().rename(
                                     columns={'visitors': 'median_visitors'})
    mainTable = pd.merge(mainTable, temp, how='left', 
                         on=['air_store_id', 'target_weekday'])

    mainTable['visitors'] = mainTable['visitors'].apply(lambda x: np.log1p(x))

    # Save produced table to disk
    mainTable.to_csv('output/main_tbl.csv')

    return mainTable


# %% Plotting

def plot_corr(size=8):
    """ Plot the correlation between features"""
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'reserve_visitors', 'test', 'visitors']
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
# ## Modelling attempts


def SGDFit():
    """ Fit using stochastic gradient descent """
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'reserve_visitors', 'test', 'visitors', 'min_visitors',
                   'max_visitors', 'median_visitors']
    X = model_data[target_cols]
    
    target_cols_fit = [col for col in X.columns if col not in ['test',
                                                               'visitors']]
    Xsub = X[X['test'] == 1]
    Xsub = Xsub.drop(['test'], axis=1)
    X = X[X['test'] == 0]
    X = X.drop(['test'], axis=1)
    

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', SGDRegressor())])
    parameters = {'clf__alpha': np.logspace(-4, 1, 6)}
    pipe = GridSearchCV(pipe, parameters)
    pipe.fit(X[target_cols_fit], X['visitors'])
    print(pipe.best_params_)

    # Predictions for submission
    Xsub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    SGD_sub = pd.read_csv('submissions/baseline.csv')
    SGD_sub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    SGD_sub['visitors'] = SGD_sub['visitors'].apply(lambda x: np.expm1(x))
    SGD_sub.to_csv('submissions/SGD.csv', index=False)
    
    return SGD_sub

def ensembleFit():
    """ Fit using gradient boosted regression """
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'reserve_visitors', 'test', 'visitors']
    X = model_data[target_cols]
    target_cols_fit = [col for col in X.columns if col not in ['test',
                                                               'visitors']]
    Xsub = X[X['test'] == 1]
    Xsub = Xsub.drop(['test'], axis=1)
    X = X[X['test'] == 0]
    X = X.drop(['test'], axis=1)

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', GradientBoostingRegressor(verbose=1))])
    parameters = {'clf__max_depth': np.linspace(1, 6, 6),
                  'clf__learning_rate': np.logspace(-3, 2, 6)}
    pipe = GridSearchCV(pipe, parameters)
    pipe.fit(X[target_cols_fit], X['visitors'])
    print(pipe.feature_importances_)
    print(pipe.loss_)

    # Predictions for submission

    Xsub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    ensembleSub = pd.read_csv('submissions/baseline.csv')
    ensembleSub.visitors = pipe.predict(Xsub[target_cols_fit])
    ensembleSub.to_csv('submissions/ensemble.csv', index=False)
    
    return ensembleSub

def NNFit():
    """ Fit using a neural network """
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'reserve_visitors', 'test', 'visitors']
    X = model_data[target_cols]
    target_cols_fit = [col for col in X.columns if col not in ['test',
                                                               'visitors']]
    Xsub = X[X['test'] == 1]
    Xsub = Xsub.drop(['test'], axis=1)
    X = X[X['test'] == 0]
    X = X.drop(['test'], axis=1)

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', MLPRegressor(verbose=1))])
    parameters = {'clf__alpha': np.logspace(-5, 2, 7)}
    pipe = GridSearchCV(pipe, parameters)
    pipe.fit(X[target_cols_fit], X['visitors'])

    # Predictions for submission

    Xsub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    NNSub = pd.read_csv('submissions/baseline.csv')
    NNSub.visitors = pipe.predict(Xsub[target_cols_fit])
    NNSub.to_csv('submissions/NN.csv', index=False)
    
    return NNSub

def XGBoost():
    """ Fit using XGBoosts """
    model_data = pd.read_csv('output/main_tbl.csv')
    target_cols = ['target_day', 'target_month', 'target_weekday',
                   'target_year', 'latitude', 'longitude', 'holiday_flg',
                   'reserve_visitors', 'test', 'visitors', 'min_visitors',
                   'max_visitors', 'median_visitors']
    X = model_data[target_cols]
    
    target_cols_fit = [col for col in X.columns if col not in ['test',
                                                               'visitors']]
    Xsub = X[X['test'] == 1]
    Xsub = Xsub.drop(['test'], axis=1)
    X = X[X['test'] == 0]
    X = X.drop(['test'], axis=1)
    

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', xgb.XGBRegressor())])
    parameters = {'clf__max_depth': np.linspace(2, 8, 7).astype(int),
                  'clf__learning_rate': np.logspace(-4, -1, 4)}
    pipe = GridSearchCV(pipe, parameters)
    pipe.fit(X[target_cols_fit], X['visitors'])
    print(pipe.best_params_)

    # Predictions for submission
    Xsub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    XGBoost_sub = pd.read_csv('submissions/baseline.csv')
    XGBoost_sub['visitors'] = pipe.predict(Xsub[target_cols_fit])
    XGBoost_sub['visitors'] = XGBoost_sub['visitors'].apply(lambda x: np.expm1(x))
    XGBoost_sub.to_csv('submissions/XGBoostGR.csv', index=False)
    
    return XGBoost_sub


SGDTable = SGDPreprocessing(air_reserve, air_store_info, air_visit_data,
                            hpg_reserve, hpg_store_info, date_info,
                            sample_submission, store_id_relation)

# SGD_sub = SGDFit()
# NNSub = NNFit()
# ensembleSub = ensembleFit()
XGBoostSub =  XGBoost()
