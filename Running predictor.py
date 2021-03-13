# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:31:59 2021

@author: manue
"""

import numpy as np  # useful for many scientific computing in Python
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import datasets, linear_model

#import data
df = pd.read_csv (r'C:\Users\manue\Documents\10k python\Data_running.csv')


#funtions to adjust time
def get_secfromhour(time_str):
    h, m, s = time_str.split(':')
    i= int(h) * 3600 + int(m) * 60 + int(s)
    return i

def get_secfrommin(time_str):
    m, s = time_str.split(':')
    i= int(m) * 60 + int(s)
    return i

#change dates to gregorian to better use regressions
def togregorian(x):
    gregoriandate = dt.date(1, 1, 1) + dt.timedelta(days=x)
    return gregoriandate

#Drops runs that were not continuos, meaning I stopped, rested and went on
indexNames = df[df['Favorite'] == True ].index #delete runs with laps
df.drop(indexNames , inplace=True)

#some preprocessomg
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date','Distance','Calories', 'Time', 'Avg Pace', 'Avg HR']]
df.replace("--", np.nan, inplace = True) #preprocess
df.dropna(inplace=True)
df.rename(columns={'Avg Pace':'Avg_Pace', 'Avg HR':'Avg_HR'}, inplace=True)
df['Calories'] = df['Calories'].str.replace(r',', '')
df["Calories"] = pd.to_numeric(df["Calories"])
df["Avg_HR"] = pd.to_numeric(df["Avg_HR"])
df['Time']=df['Time'].apply(get_secfromhour)
df['Avg_Pace']=df['Avg_Pace'].apply(get_secfrommin)
df['DateOrdinal'] = pd.to_datetime(df['Date'])
df['DateOrdinal']=df['DateOrdinal'].map(dt.datetime.toordinal)

sns.regplot(x="Date", y="Avg_Pace", data=df).set_title('Avg_Pace (secs) per Date')
plt.show() 


    #%% #plotting and regression

sns.regplot(x="DateOrdinal", y="Avg_Pace", data=df).set_title('Regression Avg_Pace (secs) per Date')
plt.show()    
    
DateYouWant = dt.date(2021, 9, 22)
Date_In_Ordinal = DateYouWant.toordinal()

X_train = df[['Distance','DateOrdinal','Avg_HR']].values
y_train = df['Avg_Pace']
X_test = np.array([10, Date_In_Ordinal, 175]).reshape(1,-1)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print("the value will be: {}".format(y_pred[0]))

Datelist=[]
Predlist=[]

for i in range(3,13):
    DateYouWant = dt.date(2021, i, 22)
    Datelist.append(DateYouWant)
    Date_In_Ordinal = DateYouWant.toordinal()
    X_train = df[['Distance','DateOrdinal','Avg_HR']].values
    y_train = df['Avg_Pace']
    X_test = np.array([10, Date_In_Ordinal, 175]).reshape(1,-1)
    
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    y_pred = int(y_pred[0])
    Predlist.append(y_pred)
    tenKtime = y_pred*10
    tenKtime = str(dt.timedelta(seconds=tenKtime))
    y_pred = str(dt.timedelta(seconds=y_pred))
    print("In month {} of 2021 you will run 10k at a pace of: {} totaling {}".format(i, y_pred, tenKtime))
    
    
    #%% Companisating 4 % for "race effect"
    
for i in range(3,13):
    DateYouWant = dt.date(2021, i, 30)
    Date_In_Ordinal = DateYouWant.toordinal()
    X_train = df[['Distance','DateOrdinal','Avg_HR']].values
    y_train = df['Avg_Pace']
    X_test = np.array([10, Date_In_Ordinal, 175]).reshape(1,-1)
    
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    y_pred = int(y_pred[0])*0.94
    tenKtime = y_pred*10
    tenKtime = str(dt.timedelta(seconds=tenKtime))
    y_pred = str(dt.timedelta(seconds=y_pred))
    print("In month {} of 2021 you will run 10k at a pace of: {} totaling {}".format(i, y_pred, tenKtime))
    
#%% Create a dataframe form 2020 and get some descriptive data from it

recent_frame = df[df['Date'] > pd.datetime(2020, 1, 1)]
recent_frame = recent_frame[recent_frame['Date'] < pd.datetime(2021, 1, 1)]

#distance ran per month
recent_frame.groupby([df['Date'].dt.month_name()], sort=False).sum().eval('Distance')\
  .plot(kind='bar', title='Distance ran (Km) per month 2020')
plt.show()
#run count per month
recent_frame.groupby([df['Date'].dt.month_name()], sort=False).count().eval('Distance')\
  .plot(kind='bar', title='Count of races per month 2020')
plt.show()
  


        #%% Somegraphs

df_gptest = recent_frame[['Date','Avg_Pace']]
df_gptest.reset_index(inplace=True)

df_gptest['month'] = recent_frame['Date'].dt.strftime('%b')

fig, ax = plt.subplots()
ax.set_title('Boxplots Avg_pace in 2020')
fig.set_size_inches((12,4))
sns.boxplot(x='month',y='Avg_Pace',data=df_gptest,ax=ax)
plt.show()

        #%% Somegraphs per week
recent_frame = recent_frame.assign(Month=recent_frame['Date'].dt.month, Week=recent_frame['Date'].dt.week, 
                                   Weekday=recent_frame['Date'].dt.weekday)


fig, axr = plt.subplots(3, figsize=(14,8), sharex=True)
recent_frame.groupby('Week')['Distance'].sum().plot.bar(ax=axr[0])
axr[0].set_title('Distance')
axr[0].set_ylabel('Km')
recent_frame.boxplot(['Avg_Pace'], by='Week', ax=axr[1])
axr[1].set_ylabel('Avg_Pace')
recent_frame.groupby('Week')['Avg_Pace'].count().plot.bar(ax=axr[2], color='C1')
axr[2].set_title('Number of trainings')
plt.show()

        #%% Somegraphs per weekday

weekdays = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig, axr = plt.subplots(3, figsize=(14,10), sharex=True)
recent_frame.groupby('Weekday')['Distance'].sum().plot.area(ax=axr[0])
axr[0].set_title('Total number of kilometers')
axr[0].set_ylabel('Km')
axr[0].set_xticklabels(weekdays)
recent_frame.boxplot(['Avg_Pace'], by='Weekday', ax=axr[1])
axr[1].set_ylabel('Km/h')
axr[1].set_xticklabels(weekdays)
recent_frame.groupby('Weekday')['Avg_Pace'].count().plot.area(ax=axr[2], color='C1')
axr[2].set_title('Number of trainings')
axr[2].set_xticklabels(weekdays)
plt.show()

        #%% Somegraphs per regression

recent_frame.reset_index()

plt.figure(figsize=(15,4))
plt.scatter(x=recent_frame.Date.values, y=recent_frame.Distance, c=recent_frame.Avg_Pace, s=recent_frame.Avg_Pace, label=None)
plt.colorbar(label='Avg_Pace')
plt.xlabel('Date')
plt.ylabel('Distance of run')
plt.legend(scatterpoints=1, title='Pace', labelspacing=1.2)
plt.show()


        #%% pie chart

import calmap

# Import Data
recent_frame.to_csv(r'C:\Users\manue\Documents\10k python\recent_frame.csv')
recent_frame = pd.read_csv (r'C:\Users\manue\Documents\10k python\recent_frame.csv', parse_dates=['Date'])

recent_frame.set_index('Date', inplace=True)

# Plot
plt.figure(figsize=(32,20), dpi= 80)
calmap.calendarplot(recent_frame['2020']['Distance'], fig_kws={'figsize': (32,20)}, yearlabel_kws={'color':'black', 'fontsize':28}, subplot_kws={'title':'My running 2020'})
plt.show()

