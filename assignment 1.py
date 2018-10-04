#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:27:51 2018

@author: demaggis
"""
#necessary modules
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------------------------------------------------

# Numpy exercises

#Ex 1 -> inner product

x = np.random.rand(10)
y = np.random.rand(10)

# manual solutions
x.dot(y)
x.T @ y #trated as matrices
np.sum(x.T*y)
np.sum(x.T*y)

# using arrays
k = len(x)
S = np.empty((k)) # random array with length=k=10
for i in range(k):
    S[i] = x.T[i]*y[i]
np.sum(S)

# using numpy --> used as a counterproof
np.inner(x,y)

# ---------------------------------------------------------------------------------------------------------------------

#Ex 2 -> mean absolute error

U = np.empty(k) #array

for i in range(k):
    U[i] = np.abs(y[i]-x[i])

MAE = (np.sum(U))/len(x)
MAE

#counterproof
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, x) # ok

# ---------------------------------------------------------------------------------------------------------------------

#Ex 3 -> lead & lag

n=2
x = np.r_[1:6]

def lead(x,n):
    S = np.empty((len(x)))
    for i in range(1,n+1):
        S[len(x)-i]="NaN"
    for i in range(n,len(x)):
        S[i-n]=(x[i])
    return S
    
lead(x,n)    

def lag(x,n):
    S = np.empty((len(x)))
    for i in range(n):
        S[i]="NaN"
    for i in range(len(x)-n):
        S[len(x)-n+i-1]=(x[i])
    return S

lag(x,n) 

# ---------------------------------------------------------------------------------------------------------------------

#Ex 4 --> euclidean distance

X_2 = np.c_[1:4,8:11] #nxd where d=2
X_2
X_3 = np.c_[1:4,8:11,20:23] #nxd where d=3
X_3

y_2=np.r_[5,6] #with d=2
y_3=np.r_[5,6,7] #with d=3

def dist(X,y):
    A = np.empty([X.shape[0],X.shape[1]])
    A_2 = np.empty((X.shape[0]))
    for i in range(X.shape[0]):
        for l in range(X.shape[1]):
            A[i,l] = (X[i,l]-y[l])**2
        A_2[i] = np.sqrt(np.sum(A[i,]))
    return A_2

dist(X_2,y_2)
dist(X_3,y_3)

# counterproof with d=2
from scipy.spatial import distance
dist1, dist2, dist3 = distance.euclidean(X_2[0,], y_2) , distance.euclidean(X_2[1,], y_2) , distance.euclidean(X_2[2,], y_2)
dist1, dist2, dist3

# counterproof with d=3
dist4, dist5, dist6 = distance.euclidean(X_3[0,], y_3) , distance.euclidean(X_3[1,], y_3) , distance.euclidean(X_3[2,], y_3)
dist4, dist5, dist6

# ---------------------------------------------------------------------------------------------------------------------

# Pandas exercises

# Ex. 1 

data = pd.read_csv('nycflights13_weather.csv',sep=',',skiprows=42)

def conv_far_celsius(fahr):
    cels = (fahr - 32) * 5 / 9
    return cels

data.temp = conv_far_celsius(data.temp)
data.temp = data.temp.interpolate() #interpolating

# dataset with origin=JFK
data_JFK = data[data.origin=="JFK"]

# daily mean temp
daily_mean_temp = []
data_JFK_with_new_index = data_JFK.set_index('day')

for i in range(1,13):
    monthly_data = data_JFK_with_new_index[data_JFK_with_new_index.month == i]
    if monthly_data.iloc[len(monthly_data)-1].name < 29:
        for a in range(1,29):
            daily_mean_temp.append([monthly_data.temp.loc[a].mean(),a,monthly_data.month.loc[a].iloc[1],monthly_data.year.loc[a].iloc[1]])
    elif monthly_data.iloc[len(monthly_data)-1].name < 30:
        for a in range(1,30):
            daily_mean_temp.append([monthly_data.temp.loc[a].mean(),a,monthly_data.month.loc[a].iloc[1],monthly_data.year.loc[a].iloc[1]])
    elif monthly_data.iloc[len(monthly_data)-1].name < 31:
        for a in range(1,31):
            daily_mean_temp.append([monthly_data.temp.loc[a].mean(),a,monthly_data.month.loc[a].iloc[1],monthly_data.year.loc[a].iloc[1]])
    else:
        for a in range(1,32):
            daily_mean_temp.append([monthly_data.temp.loc[a].mean(),a,monthly_data.month.loc[a].iloc[1],monthly_data.year.loc[a].iloc[1]])

daily_mean_temp_df = pd.DataFrame(data=daily_mean_temp,columns=['temp','day','month','year'])            

#Alternatively: we can use for loops (using numpy) --> EXTRA 
x=[]
mean2=[]
JFK_temp_days = np.c_[data_JFK.day.values,data_JFK.temp.values]

for i in range(1,data_JFK.shape[0]):
    a=0
    if JFK_temp_days[i-1,0]==JFK_temp_days[i,0]:
        if i == (data_JFK.shape[0]-1):
            x.append(JFK_temp_days[i-1,1])
            x.append(JFK_temp_days[i,1])
            mean2.append(np.mean(x))
        else:
            x.append(JFK_temp_days[i-1,1])
    else:
        x.append(JFK_temp_days[i-1,1])
        mean2.append(np.mean(x))
        x=[]

#Plotting with Pandas
        
daily_mean_temp_df['period'] = daily_mean_temp_df.day.astype(str).str.cat(daily_mean_temp_df.month.astype(str), sep='/').str.cat(daily_mean_temp_df.year.astype(str), sep='/')

# setting ticks
ticks = []
for i in range(0,len(daily_mean_temp_df),30):
    ticks.append(i)
    
fig, ax = plt.subplots()
ax.plot(daily_mean_temp_df['period'],daily_mean_temp_df['temp'])
ax.set(xlabel='time (days)', ylabel='temperature (Celsius)',
       title='Temperature in 2013')
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.grid()
plt.show()

#temp higher than in the preceding day
daily_mean_temp_df['higher'] = daily_mean_temp_df.temp>daily_mean_temp_df.temp.shift() #selecting the rows with temp higher than the previous one
daily_mean_temp_df_higher = daily_mean_temp_df[daily_mean_temp_df['higher']==True]
daily_mean_temp_df_higher
        
# 5 hottest days
max_five_temp = daily_mean_temp_df_higher.nlargest(5, 'temp')
five_hottest_days = max_five_temp['period']
five_hottest_days

# ---------------------------------------------------------------------------------------------------------------------

# Ex. 2 

data2 = pd.read_csv('nycflights13_flights.csv',sep=',',skiprows=54)

#finding positions
pos_year = data2.columns.get_loc("year")
pos_day = data2.columns.get_loc("day")

#col between year and day 
if pos_year <  pos_day:                    #when 'year' column is before 'day' column
    col_range = [range(pos_year,(pos_day+1))]
    range_year_day = data2[data2.columns[col_range]]
    #col outside year and day
    range_outside_year_day = data2.drop(data2.columns[col_range], axis=1)
else:                                       #when 'day' column is before 'year' column
    col_range = [range(pos_day,(pos_year+1))]
    range_year_day = data2[data2.columns[col_range]]
    #col outside day and year
    range_outside_year_day = data2.drop(data2.columns[col_range], axis=1)

range_year_day , range_outside_year_day

# ---------------------------------------------------------------------------------------------------------------------

# Ex. 3

A = pd.read_csv('some_birth_dates1.csv',sep=',')
B = pd.read_csv('some_birth_dates2.csv',sep=',')
C = pd.read_csv('some_birth_dates3.csv',sep=',')

A_name_indexed = A.set_index('Name')
B_name_indexed = B.set_index('Name')
C_name_indexed = C.set_index('Name')

A_name_indexed , B_name_indexed , C_name_indexed


AB_union = pd.merge(A_name_indexed,B_name_indexed,on='Name',how='outer')
ABC_union = pd.merge(AB_union,C_name_indexed, on='Name',how='outer')
AB_inter = pd.merge(A_name_indexed,B_name_indexed,on='Name',how='inner')
AC_inter = pd.merge(A_name_indexed,C_name_indexed,on='Name',how='inner')

AB_union , ABC_union , AB_inter , AC_inter

df_AB = pd.concat([A_name_indexed, B_name_indexed]) #concatenating the dataframes
AB_without_duplicates = df_AB.drop_duplicates(keep=False) #removing the rows in common
AB_diff = pd.merge(AB_without_duplicates,A,how='inner') #isolating the ones belonging to A

AB_diff
