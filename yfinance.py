import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extract closing prices
close_prices = data['Close'].values.reshape(-1, 1)

# Scale the closing prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 60  # Define the number of time steps (adjust as needed)
x_train, y_train = prepare_data(close_prices_scaled, n_steps)

# Reshape the input data to fit the LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=10, batch_size=32)


