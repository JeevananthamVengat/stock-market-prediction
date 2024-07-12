
#to run streamlit run app.py
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model("D:\Stock-Price-Prediction-main\Stock Predictions Model.keras")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2015-01-01'
end = '2024-03-28'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(ma_50_days, 'r')
ax1.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(ma_50_days, 'r')
ax2.plot(ma_100_days, 'b')
ax2.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.plot(ma_100_days, 'r')
ax3.plot(ma_200_days, 'b')
ax3.plot(data.Close, 'g')
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8,6))
ax4.plot(predict, 'r', label='Original Price')
ax4.plot(y, 'g', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
st.pyplot(fig4)

# Prediction for tomorrow's stock movement
last_100_days = data.Close.tail(100).values.reshape(-1, 1)
last_100_days_scaled = scaler.transform(last_100_days)

input_data = last_100_days_scaled.reshape(1, -1, 1)
predicted_price = model.predict(input_data)

if predicted_price[-1, 0] > last_100_days_scaled[-1, -1]:
    st.write("Prediction: Tomorrow's stock price is expected to rise.")
else:
    st.write("Prediction: Tomorrow's stock price is expected to fall.")

