# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:34:41 2019

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
import matplotlib.pyplot as plt 
import os


def build_model(n_steps,n_features):

    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def run_model(model, n_steps, steps_ahead,path_to_model,epochs=200):
    
    path_to_data='CitiPower.csv'
    X_train, y_train, X_test, y_test, mean_demand, std_demand=load_data(path_to_data, n_steps, steps_ahead)
    n_features=X_train.shape[2]
    
    if model is None:
        model=build_model(n_steps,n_features)
        try:
            history=model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=epochs)
            predicted_train = model.predict(X_train)
            predicted_test = model.predict(X_test)
            model.save(path_to_model)  # save LSTM model
        except KeyboardInterrupt:  # save model if training interrupted by user
            print('Duration of training (s) : ', time.time() - global_start_time)
            model.save(path_to_model)
            
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
    else:
        predicted_train = np.squeeze(model.predict(X_train))
        predicted_test = np.squeeze(model.predict(X_test))
    
    

    
    pred_train = predicted_train*std_demand+mean_demand
    pred_test = predicted_test*std_demand+mean_demand
 
    
    demand_train = (y_train*std_demand+mean_demand).reshape(pred_train.shape)
    demand_test=(y_test*std_demand+mean_demand).reshape(pred_test.shape)
    #Mean Absolute Percent Error on training set
    mape_train = np.mean(np.abs((demand_train-pred_train)/demand_train))*100
    #Mean Absolute Percent Error on test set
    mape_test = np.mean(np.abs((demand_test-pred_test)/demand_test))*100
    
    plt.plot(demand_test[:300],)
    plt.plot(pred_test[:300],)
    plt.title(str(steps_ahead)+' hour ahead forecast')
    plt.xlabel('Hour')
    plt.ylabel('Demand (kW)')
    plt.legend(['Observed', 'Predicted'], loc='upper left')
    plt.show()
        
    
    return pred_train, pred_test, mape_train, mape_test, demand_test
  