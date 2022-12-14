# -*- coding: utf-8 -*-
"""
Predict the open google stock price ( stock price at the beginning of a financial day)
@author: ahmed.zaalouk
"""
# import the required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the training dataset, use only the data in the "Open" column

train_dataset=pd.read_csv("Google_Stock_Price_Train.csv")
training_data= train_dataset.iloc[:,1:2].values

# Feature scailing 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_data_scaled=sc.fit_transform(training_data)


# Creating a data structure with 60 timesteps and 1 output

X_train=[]
y_train=[]

for i in range(60,len(training_data_scaled)):
    X_train.append(training_data_scaled[i-60:i,0])
    y_train.append(training_data_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

# Keras requires the input to have a specific shape for RNN.
# Therefore, we need to rehsape the input to the required shape (shape should be: [batch, timesteps, feature])
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))   

# Build the model 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model=Sequential()

# Add 1 LSTM layer with 50 units and add a dropout layer for regularization
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

# Add a second LSTM layer + Dropout
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

# Add a third LSTM layer + Dropout
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

# Add a fourth LSTM layer + Dropout
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Add an output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mse')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

#  Now lets predict the stock prices in the test set

# load the test set

test_dataset=pd.read_csv("Google_Stock_Price_Test.csv")
true_data= test_dataset.iloc[:,1:2].values

full_dataset=pd.concat((train_dataset["Open"],test_dataset["Open"]),axis=0)
input_data=full_dataset[len(full_dataset)-len(test_dataset)-60 :].values
input_data=input_data.reshape(-1,1)
# Feature scailing 
input_data_scaled=sc.transform(input_data)
#  Change the shape of the input to the required one by Keras
X_test=[]
for i in range(60,len(input_data_scaled)):
    X_test.append(input_data_scaled[i-60:i,0])
 
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1)) 

# Predict the stock prices

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot the predicted stock values vs the real stock values

# Visualising the results
plt.plot(true_data, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



