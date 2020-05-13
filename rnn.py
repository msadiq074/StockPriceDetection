# -*- coding: utf-8 -*-
"""
Created on Tue May 12 02:44:05 2020

@author: msadi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the data set
dataset = pd.read_csv('BAJFINANCE.NS.csv')
dataset.dropna(subset=["Open"],inplace=True)
dataset_test=dataset[len(dataset)-120:]
dataset_train=dataset[:len(dataset)-120]

training_set = dataset_train.iloc[:,1:2].values

# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
X_train=[] 
Y_train=[]
for i in range(60, len(dataset_train)):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# building RNN

#import libraries
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initiliaze RNN
regressor = Sequential()

# first LSTM layer + Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(rate = 0.2))

# second LSTM layer + dropout
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(rate = 0.2))

# third LSTM layer + dropout 
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(rate = 0.2))

# fourth LSTM layer + dropout
regressor.add(LSTM(units = 50,return_sequences=False))
regressor.add(Dropout(rate = 0.2))

# adding the output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')

# fitting the RNN
regressor.fit( X_train, Y_train, batch_size = 32, epochs = 100)

# Making the predictions and visualizing the results

# get real stock price
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test=[]
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_google_stock_price=regressor.predict(X_test)
predicted_google_stock_price=sc.inverse_transform(predicted_google_stock_price)

# visualizing the results
plt.plot(real_stock_price, color = 'red',label = 'Real Stock Price')
plt.plot(predicted_google_stock_price, color = 'blue',label = 'predicted Stock Price')
plt.title('Real Vs Predicted stock prices')
plt.xlabel('TIME')
plt.ylabel('STOCK PRICE')
plt.legend()
plt.show()

# Prediction for tomorrow
dataset = pd.read_csv('MARICO.NS.csv')
dataset.dropna(subset=["Open"],inplace=True)
X_test = dataset[len(dataset)-60:]
real_stock_price = X_test.iloc[:,1:2].values

# getting the predicted stock price
#dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

#inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs= real_stock_price
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test=[]
X_test=inputs
X_test = np.array(X_test)

X_test = np.reshape(X_test, (1,60,1))

predicted_google_stock_price=regressor.predict(X_test)
predicted_google_stock_price=sc.inverse_transform(predicted_google_stock_price)