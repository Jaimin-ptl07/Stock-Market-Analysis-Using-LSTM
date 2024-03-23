import numpy as np
import pandas as pd
import pandas_datareader as data

import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import streamlit as st


start = '2002-09-01'
end = '2022-07-24'

st.title('BUSSINESS ANALYTICS AND FORECASTING')

user_input = st.text_input('Enter Stoke Ticker','AMZN')
df = data.DataReader(user_input,'yahoo',start,end)

#Describing Data

st.subheader('Amazon Stock Analytics And Forecasting')
st.write(df.describe())

#Viusalization

st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index ,y='Close', data=df)
st.pyplot(fig)

st.subheader('Volume Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index ,y='Volume', data=df)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With 100 & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#load my model
model = load_model('stock_prediction_model1.h5')


# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
st.subheader('Closing Price Vs Prediction')
fig1 = plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig1)


#Bussiness analystics and forecasting
st.subheader('Amazon Bussiness Analystics And Forecasting')

import matplotlib as mp
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
  
df['Date'] = pd.to_datetime(df['Date'])

#data visulization
st.subheader('Analytics of different Fundamental Ratios')
fig = plt.figure(figsize = (14,6))
# fist line: 
plt.subplot(3,1,1)

plt.plot( df.PE_Ratio,data=df,color="red", alpha=0.4)
plt.title("PE RATIO")
plt.subplots_adjust(hspace=0.6)

# second line
plt.subplot(3,1,2)
plt.plot( df.Stock_Price, data=df, color="orange", alpha=0.3)
plt.title("STOCK PRICE")
plt.subplots_adjust(hspace=0.6)

# third line
plt.subplot(3,1,3)
plt.plot( df.TTM_Net_EPS, data=df, color="blue", alpha=0.2)
plt.title("TTM EPS")
plt.subplots_adjust(hspace=0.6)
st.pyplot(fig)

df_input = df[['Date','PE_Ratio','Stock_Price','TTM_Net_EPS']]
df_input = df_input.set_index('Date')

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input)

features = data_scaled
target = data_scaled[:,0]

TimeseriesGenerator(features, target, length=1 , sampling_rate=1, batch_size=1)[0]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)

win_length = 1
batch_size = 10
num_features = 3
train_generator = TimeseriesGenerator(x_train,y_train,length=win_length,sampling_rate=1,batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test,y_test,length=win_length,sampling_rate=1,batch_size=batch_size)

#loading model
model = load_model('amazone_prediction.h5')

model.evaluate_generator(test_generator, verbose=0)
prediction=model.predict_generator(test_generator)

x_test[:,1:][win_length:]
df_pred = pd.concat([pd.DataFrame(prediction), pd.DataFrame(x_test[:,1:][win_length:])],axis=1)

rev_transform = scaler.inverse_transform(df_pred)

df_final = df_input[prediction.shape[0]*-1:]
df_final['predicted']=rev_transform[:,0]

#data visulization
st.subheader('Forecast of different Fundamental Ratios')
fig4 = plt.figure(figsize=(12,6))
plt.plot(df_final['PE_Ratio'],'r')
plt.plot(df_final['predicted'],'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)