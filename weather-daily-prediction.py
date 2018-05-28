#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:23:46 2018

@author: Carlos
"""
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import redis
import pandas as pd
# to save the categories to file
import pickle
from numpy import concatenate
from keras.models import load_model
from email.mime.text import MIMEText
from subprocess import Popen, PIPE

# Choose variables
variables = ['rainlevel', 'time', 'datetime', 'drytemperature', 'humidity', 'heatindex', 'pressure', 'lightindex', 'temperature']

# Dataset path and file name
dsPath = './'
dsMinMaxFileName = 'weather-prediction-minmax.pkl'
dsModelFileName = 'weather-prediction-tf1.h5'

dsMinMaxPathFileName = dsPath + dsMinMaxFileName
dsModelPathFileName = dsPath + dsModelFileName

# create a connection to the localhost Redis server instance, by
# default it runs on port 6379
print('Connecting to redis')
redis_db = redis.StrictRedis(host="192.168.1.103", port=6379, db=0)
features = []
records = []

minutes = 61
last_key = int(redis_db.get('weathercollid'))-minutes
features = []
records = []
for minute in range(minutes):
    curr_key = last_key + minute
    for key in redis_db.keys('weather:'+str(curr_key)):
        features = [item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).keys())]
        records.append([item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).values())])
print('Preparing data')
df0 = pd.DataFrame(records, columns=features)
df0 = df0.drop('date', axis=1)
for column in df0.columns:
    if column != 'time':
        df0[column] = df0[column].astype(float)

df0.isnull().sum()
df0 = df0.fillna(method='ffill')
df0.isnull().sum()
df0.describe()

df0['time'] = (df0['time'].astype(float)/3600).astype(int)*3600
df0['datetime'] = pd.to_datetime(df0['time'], unit='s')
df0 = df0.groupby(['datetime']).mean()
df0.drop('time', axis=1)
variables = ['rainlevel', 'drytemperature', 'humidity', 'heatindex', 'pressure', 'lightindex', 'temperature']
df0 = df0[variables]
# summarize first 5 rows
values = df0.values

# ensure all data is float
values = values.astype('float32')[[-1], :]
# normalize features
print('normalize features')
scalefit = pickle.load(open(dsMinMaxPathFileName, 'rb'))

test = scalefit.transform(values)

test = test.reshape((test.shape[0], 1, test.shape[1]))

# Load model of file serialized
print('loading model')
model = load_model(dsModelPathFileName)
print('model loaded')

# make a prediction
print('make a prediction')
yhat = model.predict(test)
test = test.reshape((test.shape[0], test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test[:, 1:]), axis=1)
inv_yhat = scalefit.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
last_key = int(redis_db.get('weathercollid'))
key = redis_db.keys('weather:'+str(last_key))[0].decode('utf8')
inv_y = float(redis_db.hgetall(key).get(b'rainlevel').decode('utf8'))
date_y = redis_db.hgetall(key).get(b'date').decode('utf8')
# calculate RMSE
error = abs(inv_y - inv_yhat)

line = 'Predicted RainLevel: {} '.format(int(inv_yhat[0]))
print(line)
message = line + "\n"

line = 'Collected at: {} RainLevel: {}'.format(date_y, int(inv_y)) 
print(line)
message = message + line + "\n"

line = 'Test Error: %.3f' % error
print(line)
message = message + line + "\n"

if error > 300 or int(inv_yhat[0]) < 600:
    msg = MIMEText(message, "html")
    msg["From"] = "czgrqg@gmail.com"
    msg["To"] = "czgrqg@gmail.com"
    msg["Subject"] = "Email de teste"
    p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE, universal_newlines=True)
    p.communicate(msg.as_string())
