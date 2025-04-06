import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Streamlit App Config
st.set_page_config(page_title="Enhanced Stock Forecast App", layout="centered")
st.title("📈 Enhanced Stock Price Forecast App")

# Load valid tickers
try:
    ticker_df = pd.read_csv("valid_nse_tickers.csv")
    ticker_dict = dict(zip(ticker_df['Stock Name'], ticker_df['Ticker']))
except Exception as e:
    st.error(f"Error loading tickers: {e}")
    st.stop()

# Dropdown for ticker selection
selected_company = st.selectbox("Select a stock:", list(ticker_dict.keys()))
ticker = ticker_dict[selected_company]

# Forecast days input
forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

# Fetch historical data
df = yf.download(ticker, period="5y", auto_adjust=True)
if df.empty:
    st.error(f"No data found for ticker '{ticker}'.")
    st.stop()

# Load the pre-trained model or train on the fly
model_path = f"models/{ticker}_sentimentLSTM.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.info("Training model for the selected stock...")
    from model import build_model, create_training_sequences
    X, y = create_training_sequences(df[['Close']].values)
    model = build_model((60, 1))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
    model.save(model_path)

# Prepare input for forecasting
data = df[['Close']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
last_sequence = data_scaled[-60:]
X_input = last_sequence.reshape((1, 60, 1))

# Generate forecast
forecast_scaled = []
for _ in range(forecast_days):
    next_pred = model.predict(X_input)[0][0]
    forecast_scaled.append(next_pred)
    new_entry = np.array([[next_pred]]).reshape((1, 1, 1))
    X_input = np.append(X_input[:, 1:, :], new_entry, axis=1)
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

# Plotting the forecast
st.subheader("📉 Forecasted Close Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pd.date_range(start=df.index[-1], periods=forecast_days), forecast, color='orange', label="Forecast")
ax.set_title(f"{ticker} Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Display forecast table
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=df.index[-1], periods=forecast_days),
    'Forecasted Close': forecast.flatten()
})
st.dataframe(forecast_df)
