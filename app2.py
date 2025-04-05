import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils import custom_object_scope
import os

# --- Streamlit App Config ---
try:
    ticker_df = pd.read_csv(r"C:\GIT REPOS\Final-Sem-Project\DATA\valid_nse_tickers.csv")  # Ensure this CSV has 'Company' and 'Ticker' columns
    ticker_dict = dict(zip(ticker_df['Stock Name'], ticker_df['Ticker']))
except Exception as e:
    st.error(f"Error loading tickers: {e}")
    st.stop()

# NSE Holidays for 2025 (Indian financial year)
indian_holidays_2025 = pd.to_datetime([
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18",
    "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25"
])

# Streamlit UI
st.set_page_config(page_title="Stock Forecast App", layout="centered")
st.title("📈 Stock Price Forecast App")

# Dropdown for ticker selection
selected_company = st.selectbox("Select a stock:", list(ticker_dict.keys()))
ticker = ticker_dict[selected_company]

# Forecast days input
forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)
try:
    df = yf.download(ticker, period="5y", auto_adjust=True)
    if df.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data for ticker '{ticker}': {e}")
    st.stop()

if 'Close' not in df.columns:
    st.error("'Close' column not found in the dataset.")
    st.stop()

# --- Preprocess Data ---
data = df[['Close']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --- Prepare Last Window for Prediction ---
sequence_length = 60
if len(data_scaled) < sequence_length:
    st.error("Not enough historical data to make predictions. Minimum 60 data points required.")
    st.stop()

last_sequence = data_scaled[-sequence_length:]
X_input = np.reshape(last_sequence, (1, sequence_length, 1))

# --- Load Model Based on Ticker ---
if ".NS" in ticker:
    model_path = r"C:\GIT REPOS\Final-Sem-Project\TEST\2010\MODEL.h5"
else:
    model_path = r"C:\GIT REPOS\Final-Sem-Project\TEST\2010\^NSEI.h5"

try:
    with custom_object_scope({}):
        model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model from {model_path}: {e}")
    st.stop()

# --- Indian Stock Market Holidays for FY 2025 ---
# NSE Holidays for 2025 (from image)
indian_holidays_2025 = pd.to_datetime([
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18",
    "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25"
])

indian_holidays_2025 = pd.to_datetime(indian_holidays_2025)

# --- Generate Future Trading Dates ---
last_date = df.index[-1]  # Get last available date in dataset
future_dates = []
current_date = last_date + pd.Timedelta(days=1)

while len(future_dates) < forecast_days:
    if current_date.weekday() < 5 and current_date not in indian_holidays_2025:
        future_dates.append(current_date)
    current_date += pd.Timedelta(days=1)

# --- Forecast Next `forecast_days` Days ---
forecast_scaled = []
input_seq = X_input.copy()

for _ in range(forecast_days):
    next_pred = model.predict(input_seq, verbose=0)[0][0]
    forecast_scaled.append(next_pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

# --- Canvas-Style Forecast Plot ---
st.subheader("📉 Forecasted Close Prices")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(future_dates, forecast, color='orange', linewidth=2, marker='o', label="Forecasted Close")

ax.set_title(f"{ticker} Forecast - Next {forecast_days} Trading Days", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_facecolor("#f8f9fa")  # Light canvas background
fig.patch.set_facecolor('#f8f9fa')  # Match figure background
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# --- Forecast Table ---
st.subheader("Forecasted Values")
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Close': forecast.flatten()
})
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])  # Ensure proper datetime format
forecast_df.set_index('Date', inplace=True)
st.dataframe(forecast_df, use_container_width=True)

st.success("✅ Forecast complete!")
