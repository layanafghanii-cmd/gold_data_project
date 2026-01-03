# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# =============================
# تحميل الملفات
# =============================
df = pd.read_csv("gold_data_cleaned_pca.csv")  # Dataset
df['Date'] = pd.to_datetime(df['Date'])

model = load_model("gold_lstm_model.h5", compile=False)  # H5 بدون مشاكل
scaler = joblib.load("scaler.pkl")  # Scaler

# =============================
# Streamlit Interface
# =============================
st.title("Gold Price Prediction ⏱️")
st.write("أدخل التاريخ لتحصل على السعر المتوقع مع Error")

user_date = st.date_input("اختر التاريخ")

if st.button("Predict"):

    # =============================
    # تجهيز Features للـ prediction
    # =============================
    if user_date in list(df['Date'].dt.date):
        # التاريخ موجود بالداتا
        row = df[df['Date'].dt.date == user_date]
        X = row.drop(['Date', 'Target'], axis=1).values
    else:
        # التاريخ غير موجود → خذ آخر صف ك approximation
        last_row = df.drop(['Date', 'Target'], axis=1).iloc[-1].values
        X = np.array([last_row])  # شكل 2D

    # Scaling + reshape
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[1])

    # Prediction
    prediction = model.predict(X_scaled)
    predicted_price = prediction[0][0]

    # =============================
    # حساب Error على كل البيانات (اختياري)
    # =============================
    X_all = df.drop(['Date','Target'], axis=1).values
    y_all = df['Target'].values
    X_all_scaled = scaler.transform(X_all)
    X_all_scaled = X_all_scaled.reshape(X_all_scaled.shape[0],1,X_all_scaled.shape[1])
    y_pred_all = model.predict(X_all_scaled)
    mae = mean_absolute_error(y_all, y_pred_all)

    # =============================
    # عرض النتائج
    # =============================
    st.success(f"💰 السعر المتوقع: {predicted_price:.4f}")
    st.info(f"📉 MAE Error: {mae:.4f}")





