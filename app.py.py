# streamlit_lstm_gold.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Prediction LSTM", layout="wide")
st.title("📈 Gold Price Prediction Using LSTM")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file with columns: Date, Price_Gold, Volume_Gold, Change%_Gold, Price_Oil, Price_Dollar, Price_Stocks, Volume_Stocks", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # 2️⃣ Feature scaling
    features = ['Price_Gold', 'Volume_Gold', 'Change%_Gold', 'Price_Oil',
                'Price_Dollar', 'Price_Stocks', 'Volume_Stocks']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_data = pd.DataFrame(scaled_data, columns=features, index=df.index)
    
    n_days = st.slider("Select number of past days for LSTM input:", 1, 60, 30)
    
    # Prepare sequences
    X, y = [], []
    for i in range(n_days, len(scaled_data)):
        X.append(scaled_data.iloc[i-n_days:i].values)
        y.append(scaled_data.iloc[i]['Price_Gold'])
    X, y = np.array(X), np.array(y)
    
    st.write(f"Input shape (X): {X.shape}")
    st.write(f"Output shape (y): {y.shape}")
    
    # Train/test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 3️⃣ Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    st.subheader("Model Summary")
    st.text(model.summary())
    
    # 4️⃣ Train model
    epochs = st.number_input("Number of epochs:", min_value=1, max_value=200, value=50)
    batch_size = st.number_input("Batch size:", min_value=1, max_value=128, value=32)
    
    if st.button("Train LSTM Model"):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
        st.success("✅ Model trained successfully!")
        
        # 5️⃣ Predictions
        predicted = model.predict(X_test)
        
        # Inverse scaling for Price_Gold
        gold_scaler = MinMaxScaler()
        gold_scaler.min_, gold_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
        predicted_original = gold_scaler.inverse_transform(predicted)
        y_test_original = gold_scaler.inverse_transform(y_test.reshape(-1,1))
        
        # 6️⃣ Performance metrics
        mse = mean_squared_error(y_test_original, predicted_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predicted_original)
        r2 = r2_score(y_test_original, predicted_original)
        
        st.subheader("Performance Metrics for Gold Price Prediction")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # 7️⃣ Plot actual vs predicted
        st.subheader("Gold Price Prediction vs Actual")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df.index[train_size+n_days:], y_test_original, label="Actual Price", color='blue')
        ax.plot(df.index[train_size+n_days:], predicted_original, label="Predicted Price", color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Gold Price")
        ax.legend()
        st.pyplot(fig)




