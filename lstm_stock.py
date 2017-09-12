# this is to predict the stock prices of stock using the data from last 60 days
# so we predict say 61st day price, after taking into consideration the prices for the first 60 days
# the implementation is done using keras
# it can predict on the basis of multiple values as input
# it can take any number of features as input, currently 3 (open, high, low) predicting the first one
# the continuation would be to get news about stock, do sentiment analysis, and use that also to improve prediction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

window=25
noOfMetrics = 3
#Open, High, Low
# We are trying to predict Open price, on the basis of all 3
datasetTrain = pd.read_csv('Google_Stock_Price_Train.csv')
trainingSet = datasetTrain.iloc[:,1:4].values
#print(trainingSet)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)

# this is just for untransforming, taking only open values
#if this was not done, it was causing array indexes not matching
valuesToBePredicted = datasetTrain.iloc[:,1:2].values
sc1 = MinMaxScaler(feature_range = (0, 1))
trainingSetScaled1 = sc1.fit_transform(valuesToBePredicted)

# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(window, 1258):
    X_train.append(trainingSetScaled[i-window:i])
    y_train.append(trainingSetScaled[i,0])
#print(X_train)
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#(batch_size, time_steps(60), features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], noOfMetrics))
print(X_train.shape)
print(y_train.shape)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
#units basically curresponds to the no of hidden layers in the weight matrix
#regressor.add(LSTM(units=3, return_sequences = True, input_shape = (None, 1)))

# Adding a second LSTM layer
#regressor.add(LSTM(units = 256, return_sequences = True))

# Adding a fourth LSTM layer
#regressor.add(LSTM(units = 3, dropout=0.2))

# Adding the output layer
#regressor.add(Dense(units = 1))
#regressor.add(Activation('linear'))
# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (None, noOfMetrics)))
regressor.add(LSTM(units = 64))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 25, batch_size = 32)

# Getting the real stock price for February 1st 2012 - January 31st 2017
datasetTest = pd.read_csv('Google_Stock_Price_Test.csv')
testSet = datasetTest.iloc[:,1:4].values
realStockPrice = np.concatenate((trainingSet[0:1258], testSet), axis = 0)

# Getting the predicted stock price of 2017
scaled_realStockPrice = sc.fit_transform(realStockPrice)
inputs = []
for i in range(1258, 1278):
    inputs.append(scaled_realStockPrice[i-window:i])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], noOfMetrics))
predictedStockPrice = regressor.predict(inputs)
#print(predictedStockPrice)
#this is a issue, since sc expects array of size [20,3] and we have predicted values of form [20,1]
# can do this process manually to avoid this issue
predictedStockPrice = sc1.inverse_transform(predictedStockPrice)
#print(predictedStockPrice)
#only printing the values which we are predicting
plt.plot(realStockPrice[1258:,0], color = 'red', label = 'Real')
plt.plot(predictedStockPrice, color = 'blue', label = 'Predicted')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()