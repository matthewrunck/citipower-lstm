# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:17:37 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd

df = pd.read_csv('CitiPower.csv', parse_dates=[0])
demand = df['Demand (kW)']

mean_demand = np.mean(demand)
std_demand = np.std(demand)
shift_demand = (demand-mean_demand)/std_demand
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt 


# split a univariate sequence into samples for one hour ahead forecasts


def split_sequence_uni(sequence, n_steps_back,n_steps_ahead):
    N = len(sequence)-n_steps_back-n_steps_ahead
    X=np.zeros((N,n_steps_back))   
    y=np.zeros((N,n_steps_ahead))
    for i in range(len(sequence)):
   	# find the end of this pattern
       end_ix = i + n_steps_back
   	# check if we are beyond the sequence
       if end_ix > len(sequence)-n_steps_ahead-1:
           break
   		# gather input and output parts of the pattern
       X[i,:] = sequence[i:end_ix]
       y[i,:] = sequence[end_ix:end_ix+n_steps_ahead]
    return X,y
# choose a number of time steps
n_steps_back = 24
n_steps_ahead=24
# split into samples
X, y = split_sequence_uni(shift_demand, n_steps_back,n_steps_ahead)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

n_train = int(0.9*X.shape[0])
    
X_train = X[:n_train,:,:]
y_train =y[:n_train,]
    
X_test = X[n_train:,:,:]
y_test =y[n_train:,]
    
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_back, n_features)))
model.add(Dense(24))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=200, verbose=1)
# demonstrate prediction
x_input = array(shift_demand[8735:8759])
x_input = x_input.reshape((1, n_steps, n_features))
train_hat = np.array(model.predict(X_train, verbose=0))
test_hat = np.array(model.predict(X_test, verbose=0))
shift_demand = np.array(shift_demand)

train_pred = train_hat*std_demand+mean_demand
test_pred = test_hat*std_demand+mean_demand
dem_train =y_train*std_demand+mean_demand
ape_train = np.abs((dem_train-train_pred)/dem_train)*100
mape_train = np.mean(ape_train,axis=0)

dem_test =y_test*std_demand+mean_demand
ape_test = np.abs((dem_test-test_pred)/dem_test)*100
mape_test = np.mean(ape_test,axis=0)

fig = plt.figure()
plt.plot(np.arange(300),dem_train[-300:,0])
plt.plot(train_pred[-300:,0])
plt.xlabel('Hour')
plt.ylabel('Electricity load (kW')
plt.show()
fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')

