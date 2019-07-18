# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:34:15 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd

def load_data(path, n_steps, steps_ahead):
    df = pd.read_csv(path)
    
    N = df.shape[0]
    
    demand = np.array(df['Demand (kW)']).reshape((N,1))
    
    mean_demand = np.mean(demand)
    std_demand = np.std(demand)
    shift_demand = (demand-mean_demand)/std_demand
    
    hours_val = df['Day Hour']
    
    hour = np.eye(24)[hours_val]
    
    date = pd.to_datetime(df['Date'],format='%d/%m/%Y')
    day=date.dt.dayofweek
    
    weekday = np.eye(7)[day]
    
    temp = np.array(df['Dry Bulb Temp (celsius)']).reshape((df.shape[0],1))
    
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    shift_temp=(temp-mean_temp)/std_temp
    
    inp = np.concatenate((shift_demand,shift_temp,hour,weekday),axis=1)
    
    n_features = inp.shape[1]
    
   
    
    X,y = split_sequence(inp,n_steps, steps_ahead)
    
    
    #Split into training and testing
    
    n_train = int(0.9*X.shape[0])
    
    X_train = X[:n_train,:,:]
    y_train =y[:n_train,]
    
    X_test = X[n_train:,:,:]
    y_test =y[n_train:,]

    return X_train, y_train, X_test, y_test, mean_demand, std_demand

def split_sequence(sequence, n_steps, steps_ahead):
    X = list()
    X0 = list()
    y= list()
    for i in range(sequence.shape[0]):
    		# find the end of this pattern
        end_ix = i + n_steps
        		# check if we are beyond the sequence
        if end_ix > len(sequence)-steps_ahead:
            break
        		# gather input and output parts of the pattern
        seq_x = sequence[i+steps_ahead:end_ix+steps_ahead,:] 
        seq_y = sequence[end_ix+steps_ahead-1,0]
        seq_x0 = sequence[i:end_ix,0] 
        X.append(seq_x)
        X0.append(seq_x0)
        y.append(seq_y)
    y=np.array(y)
    X=np.array(X)
    X0=np.array(X0)
    X[:,:,0]=X0
    return X, y

