# streamlit_lstm_predict.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Prediction", layout="wide")
st.title("📈 Gold Price Prediction Using LSTM")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload CSV with columns: Date, Price_Gold, Volume_Gold, Change%_Gold, Price_Oil, Price_Dollar, Price_Stocks, Volume_Stocks", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # 2️⃣ Scaling features
    features = ['Price_Gold', 'Volume_Gold', 'Change%_Gold', 'Price_Oil',
                'Price_Dollar', 'Price_Stocks', 'Volume_Stocks']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_data = pd.DataFrame(scaled_data, columns=features, index=df.index)
    
    n_days = 30  # fixed window for LSTM
    
    # Prepare sequences
    X, y = [], []
    for i in range(n_days, len(scaled_data)):
        X.append(scaled_data.iloc[i-n_days:i].values)
        y.append(scaled_data.iloc[i]['Price_Gold'])
    X, y = np.array(X), np.array(y)
    
    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 3️⃣ Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model once
    with st.spinner("Training LSTM model..."):
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    st.success("✅ Model trained successfully!")
    
    # Prepare scaler for inverse transform of Price_Gold
    gold_scaler = MinMaxScaler()
    gold_scaler.min_, gold_scaler.scale_ = scaler.min_[0], scaler.scale_[0]  # Price_Gold is index 0

    # Predict test set for accuracy metrics
    predicted_test = model.predict(X_test)
    predicted_test_original = gold_scaler.inverse_transform(predicted_test)
    y_test_original = gold_scaler.inverse_transform(y_test.reshape(-1,1))
    
    mse = mean_squared_error(y_test_original, predicted_test_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predicted_test_original)
    r2 = r2_score(y_test_original, predicted_test_original)
    
    st.subheader("Model Accuracy on Test Set")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.4f}")
    
    # 4️⃣ User input date for prediction
    input_date = st.date_input("Enter the date to predict gold price:")
    
    # Check if date is within dataset
    if input_date in df.index:
        idx = df.index.get_loc(input_date)
        if idx >= n_days:
            input_sequence = scaled_data.iloc[idx-n_days:idx].values.reshape(1, n_days, len(features))
            predicted_price_scaled = model.predict(input_sequence)
            predicted_price = gold_scaler.inverse_transform(predicted_price_scaled)[0][0]
            st.subheader(f"Predicted Gold Price on {input_date}: ${predicted_price:.2f}")
        else:
            st.warning(f"Not enough past {n_days} days to predict this date.")
    else:
        st.warning("Date not found in the dataset.")





