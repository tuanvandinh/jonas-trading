import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import pytz
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pandas.plotting import register_matplotlib_converters

mt5.initialize()

login = 42497659
password = 'y7g&6+L0CTK7V3'
server = 'AdmiralMarkets-Demo'
symbol = 'EURUSD'
timezone = pytz.timezone("Etc/UTC")
startdate = datetime(2024, 1, 1, tzinfo=timezone)
enddate = datetime(2024, 2, 19, tzinfo=timezone)

mt5.login(login, password, server)

rates = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, startdate, enddate))
rates['time'] = pd.to_datetime(rates['time'], unit='s')
account_info = mt5.account_info()
riskmoney = (account_info.balance / 300)

fig, ax = plt.subplots()
ax.plot(rates['time'], rates['open'], label=symbol + ' open')
ax.plot(rates['time'], rates['close'], label=symbol + ' close')
ax.legend()
plt.show()

print(account_info)
print(riskmoney)
print(rates.head())

# 'close'-Preise normalisieren
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(rates['close'].values.reshape(-1, 1))

# Trainingsdatensatz erstellen
sequence_length = 50  # Adjusted sequence length
train_size = int(len(scaled_prices) * 0.80)
train_data = scaled_prices[:train_size]

# Sequenzen und Labels erstellen
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, sequence_length)

# Daten für LSTM umformen
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM-Modell erstellen
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# LSTM-Modell trainieren
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Nun ist das Modell auf die angegebene Zeit trainiert.
# Du kannst es nun für Vorhersagen oder weiteres Training verwenden.
# Wenn Echtzeitdaten verfügbar sind, musst du den Code anpassen, um das Modell kontinuierlich zu aktualisieren.

# Verbindung trennen
mt5.shutdown()
