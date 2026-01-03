# streamlit_lstm_forecast.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast Using LSTM")

# 1️⃣ Load dataset (local or GitHub)
@st.cache_data
def load_data():
    df = pd.read_csv("dropped_corr.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    return df

df = load_data()
st.subheader("Data Preview")
st.dataframe(df.tail(10))  # آخر 10 أيام

# 2️⃣ Prepare features
features = ['Price_Gold', 'Volume_Gold', 'Change%_Gold', 'Price_Oil',
            'Price_Dollar', 'Price_Stocks', 'Volume_Stocks']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])
scaled_data = pd.DataFrame(scaled_data, columns=features, index=df.index)

n_days = 30  # LSTM window

# Prepare sequences for training
X, y = [], []
for i in range(n_days, len(scaled_data)):
    X.append(scaled_data.iloc[i-n_days:i].values)
    y.append(scaled_data.iloc[i]['Price_Gold'])
X, y = np.array(X), np.array(y)

# Train/test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3️⃣ Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
st.success("✅ Model trained successfully!")

# 4️⃣ Prepare scaler for Price_Gold
gold_scaler = MinMaxScaler()
gold_scaler.min_, gold_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

# 5️⃣ Forecast function
def forecast_price(target_date):
    last_date = df.index[-1]
    delta_days = (target_date - last_date).days
    if delta_days < 1:
        return "Please enter a future date after the last date in dataset."
    
    recent_data = scaled_data[-n_days:].values.tolist()
    
    for _ in range(delta_days):
        x_input = np.array(recent_data[-n_days:]).reshape(1, n_days, len(features))
        pred_scaled = model.predict(x_input, verbose=0)
        # نضيف predicted price مع باقي الميزات ك copy (يمكننا وضع نفس القيم لغير Price_Gold)
        new_row = recent_data[-1].copy()
        new_row[0] = pred_scaled[0][0]  # تحديث Price_Gold
        recent_data.append(new_row)
    
    final_pred_scaled = recent_data[-1][0]
    final_pred = gold_scaler.inverse_transform([[final_pred_scaled]])[0][0]
    return final_pred

# 6️⃣ User input
input_date = st.date_input("Enter a future date to predict gold price:")

if input_date > df.index[-1]:
    predicted_price = forecast_price(input_date)
    st.subheader(f"Predicted Gold Price on {input_date}: ${predicted_price:.2f}")
else:
    st.warning("Please select a date after the last date in the dataset.")








