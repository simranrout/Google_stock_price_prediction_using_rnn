#recurrent Neural Network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#IMPORT DATASET
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values

#FEATURESCALING ----MinMaxScaler is tool for  normalisation----
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set =sc.fit_transform(training_set)


#GET THE INPUT AND OUTPUT

x_train = training_set[0:1257]#as we dont need the last stock price so we need only 1257 stock price
y_train=training_set[1:1258]

# Reshaping (changing the format)
x_train=np.reshape(x_train,(1257, 1 , 1))
'''
number of observation-1257
timestep-1
FEATURES=1 (stock price at time t)
reference is KERAS DOCUMENTATION
'''
#BUILDING RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#----INTIALIZATION OF RNN
sq=Sequential()
#here we will use the concet of regression as we dont have catogerical value


#ADDING INPUT AND LSTM LAYER TO RNN
sq.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
#OUTPUT LAYER
sq.add(Dense(units= 1))#here units consist the number of layers

#compiling RNN
sq.compile(optimizer='adam',loss='mean_squared_error') #optimizer  is adam as it takes less memory,loss will not be binary (y^-y)^2
#FITTING THE RNN TO THE TRAINING SET
sq.fit(x_train,y_train,batch_size=32,epochs=200)#i got the loss of 2.4847e-04





#MAKING PREDICTION ON THE TEST SET

test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values
