# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:38:22 2019

@author: Lenovo
"""
from keras.models import load_model
import numpy as np
import pandas as pd
import datetime
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import matplotlib.pyplot as plt 
import os

n_steps = 24
retrain=False
hours_ahead = 24
mape_train_all =[]
mape_test_all =[]
pred_train_all =[]
pred_test_all =[]
demand_test_all=[]
for i in range(hours_ahead):
    steps_ahead=i+1
    
    path_to_model = os.getcwd()+'\\models\\LSTM_demand_' + str(steps_ahead)+'_hours_ahead.h5'

    if retrain:
        model=None
    else:
        try:
            model=load_model(path_to_model)
        except:
            model=None

            
    pred_train,pred_test,mape_train,mape_test, demand_test=run_model(model,n_steps,steps_ahead,path_to_model,epochs=200)
    mape_train_all.append(mape_train)
    mape_test_all.append(mape_test)
    if i>0:
        nans = np.full(i,np.nan)    
        pred_test=np.append(pred_test,nans)
    demand_test_all.append(np.squeeze(demand_test))
    pred_train_all.append(np.squeeze(pred_train))
    pred_test_all.append(np.squeeze(pred_test))
    

N=len(pred_train_all[0])+len(pred_test_all[0])
pred_toto=np.zeros((N,hours_ahead))
for i in range(hours_ahead):
    pred_toto[:,i]= np.append(pred_train_all[i],pred_test_all[i])
    

df0 = pd.read_csv('CitiPower.csv')
Date = df0['Date'][hours_ahead:]
Hour = df0['Day Hour'][hours_ahead:]
Dem = np.array(df0['Demand (kW)'])[n_steps:]

def plot_forecast_profile(pred_toto, day='1/08/2016',Hour=0):
    start = np.nonzero((Date==day) & (Hour==Hour))[0]
    fig = plt.figure()
    plt.plot(pred_toto[start,:][0])
    plt.plot(Dem[start[0]:start[0]+np.shape(pred_toto)[1]])
    plt.title('24 Hour forecast starting '+ day + '  '+str(Hour)+':00')
    plt.ylabel('Demand (kW)')
    plt.xlabel('Hour')
    plt.legend(['Predicted', 'Observed'], loc='upper left')
    fig.savefig('24_Hour_forecast_'+ day.replace('/','-') +'_'+str(Hour))

plot_forecast_profile(pred_toto, day='5/08/2016',Hour=10)

fig = plt.figure()
plt.plot(mape_train_all)
plt.plot(mape_test_all)
plt.title('Training and Test Mean Absolute Percentage Error (MAPE)')
plt.xlabel('Hours Ahead')
plt.ylabel('MAPE (%)')
plt.legend(['Training Error','Test Error'])
fig.savefig('Model Errors')


path=os.getcwd()
#plots the last 300 hours (about last two weeks of june)
def plot_n_hour_ahead_300_hours(hours_ahead):
    figpath=path+'//'+str(hours_ahead)+ '_hour_ahead _forecast.jpg'
    n=hours_ahead-1
    fig = plt.figure()
    plt.plot(Dem[-300:],)
    plt.plot(pred_toto[-300-n:-n,n],)
    plt.title(str(hours_ahead)+' hour ahead forecast')
    plt.xlabel('Hour')
    plt.ylabel('Demand (kW)')
    plt.legend(['Observed', 'Predicted'], loc='upper left')
    fig.savefig(path+'//'+str(hours_ahead)+'_hour_ahead')


for i in [1, 3, 6, 12, 18, 24]:
    plot_n_hour_ahead_300_hours(i)
