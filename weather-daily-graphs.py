#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 01:04:47 2018

@author: Carlos
"""
import redis
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt



# create a connection to the localhost Redis server instance, by
# default it runs on port 6379
redis_db = redis.StrictRedis(host="192.168.1.103", port=6379, db=0)
features = []
records = []

minutes=1441
last_key = int(redis_db.get('weathercollid')) -minutes
features = []
records = []
for minute in range(minutes):
    curr_key = last_key + minute
    for key in redis_db.keys('weather:'+str(curr_key)):
        features = [item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).keys())]
        records.append([item.decode('utf8') for item in list(redis_db.hgetall(key.decode('utf8')).values())])
df0 = pd.DataFrame(records, columns=features)
for column in df0.columns:
    if column != 'date':
        df0[column] = df0[column].astype(float)  
df0['datetime'] = pd.to_datetime(df0['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
df0['dateindex'] = df0['datetime']

df0 = df0.set_index('dateindex')
df0 = df0.sort_values('datetime')

df0.isnull().sum()
df0 = df0.fillna(method='ffill')
df0.isnull().sum()

plt.figure(figsize=(8,4))
label=['pressure']
for feature in label:
    plt.plot(df0.index, df0[feature], label=feature )
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.savefig(label[0]+'.png', dpi=200, bbox_inches='tight') 

plt.figure(figsize=(8,4))
label=['humidity']
for feature in label:
    plt.plot(df0['datetime'], df0[feature], label=feature )
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.savefig(label[0]+'.png', dpi=200, bbox_inches='tight') 

plt.figure(figsize=(8,4))
label=['rainlevel']
for feature in label:
    plt.plot(df0['datetime'], df0[feature], label=feature )
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.savefig(label[0]+'.png', dpi=200, bbox_inches='tight') 

plt.figure(figsize=(8,4))
label=['lightindex']
for feature in label:
    plt.plot(df0['datetime'], df0[feature], label=feature )
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.savefig(label[0]+'.png', dpi=200, bbox_inches='tight') 

plt.figure(figsize=(8,4))
label=['drytemperature', 'temperature', 'heatindex']
for feature in label:
    plt.plot(df0['datetime'], df0[feature], label=feature )
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.savefig(label[0]+'.png', dpi=200, bbox_inches='tight') 

 

