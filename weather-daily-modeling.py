#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:23:46 2018

@author: Carlos
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import modf
# to save the categories to file
import pickle

# Choose variables
variables = ['rainlevel', 'time', 'datetime', 'drytemperature', 'humidity', 'heatindex', 'pressure', 'lightindex', 'temperature']

# Dataset path and file name
dsPath = './'
dsInputFileName = 'weather-prediction-full.csv'
dsBackupFileName = 'weather-prediction-clean.csv'
dsMinMaxFileName = 'weather-prediction-minmax.pkl'
dsModelFileName = 'weather-prediction-tf1.h5'

dsInputPathFileName = dsPath + dsInputFileName
dsBackupPathFileName = dsPath + dsBackupFileName
dsMinMaxPathFileName = dsPath + dsMinMaxFileName
dsModelPathFileName = dsPath + dsModelFileName

# Import data
dataset = pd.read_csv(dsInputPathFileName, usecols=variables)
dataset['time'] = (dataset['time']/3600).astype(int)*3600
#dataset['datetime'] = pd.to_datetime(dataset['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
dataset['datetime'] = pd.to_datetime(dataset['time'], unit='s')
dataset = dataset.groupby(['datetime']).mean()
dataset.drop('time', axis=1)
variables = ['rainlevel', 'drytemperature', 'humidity', 'heatindex', 'pressure', 'lightindex', 'temperature']
dataset = dataset[variables]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv(dsBackupFileName)

# load dataset
dataset = pd.read_csv(dsBackupFileName, header=0, index_col=0)
values = dataset.values

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scalefit = scaler.fit(values)
pickle.dump(scalefit, open(dsMinMaxPathFileName, 'wb'))
scaled = scalefit.transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_mins = 2 * 24
train = values[:n_train_mins, :]
test = values[n_train_mins:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
tf.logging.set_verbosity(tf.logging.ERROR)
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=72, 
                    verbose=3, validation_split=0.3,
                    shuffle=False)

# Save model do file serialized
model.save(dsModelPathFileName) 

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
