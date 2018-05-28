#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import redis
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt


# create a connection to the localhost Redis server instance, by
# default it runs on port 6379
redis_db = redis.StrictRedis(host="192.168.1.103", port=6379, db=0)
features = []
records = []
for key in redis_db.keys('weather:*'):
    features = [item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).keys())]
    records.append([item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).values())])
df0 = pd.DataFrame(records, columns=features)
df0['datetime'] = pd.to_datetime(df0['time'], unit='s')
df0['dateindex'] = df0['datetime']

df0 = df0.set_index('dateindex')

df0.isnull().sum()
df0 = df0.fillna(method='ffill')
df0.isnull().sum()

df0.to_csv('weather-prediction-full.csv')
