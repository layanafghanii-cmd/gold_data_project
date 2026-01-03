import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# =============================
# تحميل الملفات
# =============================
df = pd.read_csv("gold_data_cleaned_pca.csv")
df['Date'] = pd.to_datetime(df['Date'])

from tensorflow.keras.mofels import load_modwl
model= load_model("gold_lstm_model.h5" , compile=false)
scaler = joblib.load("scaler.pkl")

st.title("Gold Price Prediction ⏱️")
st.write("أدخل التاريخ فقط لتحصل على السعر المتوقع")

# =============================
# إدخال التاريخ
# =============================
user_date = st.date_input("اختر التاريخ")

if st.button("Predict"):
    if user_date not in list(df['Date'].dt.date):
        st.error("❌ التاريخ غير موجود في الداتا")
    else:
        row = df[df['Date'].dt.date == user_date]

        X = row.drop(['Date', 'Target'], axis=1).values
        y_true = row['Target'].values[0]

        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[1])

        prediction = model.predict(X_scaled)
        predicted_price = prediction[0][0]

        # Error عام
        X_all = df.drop(['Date', 'Target'], axis=1).values
        y_all = df['Target'].values

        X_all_scaled = scaler.transform(X_all)
        X_all_scaled = X_all_scaled.reshape(X_all_scaled.shape[0], 1, X_all_scaled.shape[1])

        y_pred_all = model.predict(X_all_scaled)
        mae = mean_absolute_error(y_all, y_pred_all)

        st.success(f"💰 السعر المتوقع: {predicted_price:.4f}")

        st.info(f"📉 MAE Error: {mae:.4f}")

