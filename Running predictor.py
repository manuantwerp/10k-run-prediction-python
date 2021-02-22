# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:44:17 2021

@author: manue
"""

import numpy as np  # useful for many scientific computing in Python
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import date
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv (r'C:\Users\manue\Downloads\Activities (5).csv') #load data

#keep some
index_names = df[ (df['Distance'] != 5) & (df['Distance'] != 6)].index 
df.drop(index_names , inplace=True)

indexNames = df[df['Favorite'] == True ].index #delete runs with laps
df.drop(indexNames , inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

df.replace("--", np.nan, inplace = True) #preprocess
df.dropna(inplace=True)
df = df[['Date','Distance','Calories', 'Time', 'Avg Pace', 'Avg HR']]
df.rename(columns={'Avg Pace':'Avg_Pace', 'Avg HR':'Avg_HR'}, inplace=True)
df['Calories'] = df['Calories'].str.replace(r',', '')
df["Calories"] = pd.to_numeric(df["Calories"])
df["Avg_HR"] = pd.to_numeric(df["Avg_HR"])


def get_secfromhour(time_str):
    h, m, s = time_str.split(':')
    i= int(h) * 3600 + int(m) * 60 + int(s)
    return i

def get_secfrommin(time_str):
    m, s = time_str.split(':')
    i= int(m) * 60 + int(s)
    return i
df['Time']=df['Time'].apply(get_secfromhour)
df['Avg_Pace']=df['Avg_Pace'].apply(get_secfrommin)
'''
now some plotting
'''
def togregorian(x):
    gregoriandate = dt.date(1, 1, 1) + dt.timedelta(days=x)
    return gregoriandate

WashingtonsBirthDay = dt.date(1732, 2, 22)
gregorianOrdinal = WashingtonsBirthDay.toordinal()

df['DateOrdinal'] = pd.to_datetime(df['Date'])
df['DateOrdinal']=df['DateOrdinal'].map(dt.datetime.toordinal)
#f['Manu'] = df['DateOrdinal'].apply(togregorian)

    #%%
df.plot(x='Time', y='Calories')

sns.regplot(x="Time", y="Calories", data=df)
plt.ylim(0,)

plt.plot(df['Time'], df['Calories'], 'o', color='blue');

plt.plot(df['Date'], df['Avg_Pace'], 'o', color='blue');
    #%% modeling

Z = df[['Date', 'Distance', 'Calories', 'Avg_Pace', 'Avg_HR','DateOrdinal']]

lm = LinearRegression()
lm

X = df[['DateOrdinal']]
Y = df['Avg_Pace']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]


    #%%
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="DateOrdinal", y="Avg_Pace", data=df)
plt.ylim(0,)

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['DateOrdinal'], df['Avg_Pace'])
plt.show()

    #%%

X_train = df[['Distance','DateOrdinal']].values
y_train = df['Avg_Pace']
X_test = np.array([10, 737900]).reshape(1,-1)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Distance: {} Km, Speed: {} Km/h, Time: {} hours \n'.format(X_test[0][0], X_test[0][0]/float(y_pred[0]/60), y_pred[0]/60))
    
    #%%
X_train = df[['DateOrdinal','Avg_HR']].values
y_train = df['Avg_Pace']
Desireday = date.today().toordinal() + 90
X_test = np.array([Desireday]).reshape(1,-1)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
#print('Distance: {} Km, Speed: {} Km/h, Time: {} hours \n'.format(X_test[0], X_test[0]/float(y_pred[0]/60), y_pred[0]/60))
print(y_pred)
