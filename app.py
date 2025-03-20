import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the pre-trained model
model = load_model('stacked_lstm_model.keras')

# Function to prepare data for LSTM
def prepare_data(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Streamlit App
st.title("Stock Price Prediction using Stacked LSTM")

uploaded_file = st.file_uploader("Upload a stock price CSV file", type=["csv"])
if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df.head())

    # Ensure the data has a 'Date' and 'Close' column
    if 'Date' in df.columns and 'Close' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Select rolling average window
        roll_window = st.selectbox("Select Rolling Average Window", [5, 10, 20, 30, 50])
        df['Rolling'] = df['Close'].rolling(window=roll_window).mean()
        
        st.subheader("Historical Data with Rolling Average")
        st.line_chart(df[['Close', 'Rolling']])

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df[['Close']].values)

        time_steps = 60
        X, y = prepare_data(data_scaled, time_steps)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Prediction
        predictions = model.predict(X)
        predictions_rescaled = scaler.inverse_transform(predictions)
        actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

        # Model Evaluation
        rmse = np.sqrt(mean_squared_error(actual_rescaled, predictions_rescaled))
        mae = mean_absolute_error(actual_rescaled, predictions_rescaled)
        st.subheader("Model Performance")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")

        # Plot Predictions vs Actual
        st.subheader("Predictions vs Actual")
        plt.figure(figsize=(18, 6))
        plt.plot(actual_rescaled, label='Actual Prices')
        plt.plot(predictions_rescaled, label='Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Forecasting
        # Forecasting
# Forecasting
        forecast_steps = st.number_input("Enter Number of Steps to Forecast", min_value=1, max_value=100, value=5)
        last_sequence = np.expand_dims(data_scaled[-time_steps:], axis=0)
        forecasts = []
        
        for _ in range(forecast_steps):
            current_forecast = model.predict(last_sequence)
            forecasts.append(current_forecast[0][0])  # Assuming forecast shape is (1, 1) per step
            # Update last_sequence with the new forecast
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = current_forecast[0][0]
        
        # Rescale the forecasted values
        forecast_rescaled = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
        
        # Ensure that the length of forecast_rescaled matches the number of future dates
        future_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
        
        # Adjust DataFrame creation to avoid shape mismatch
        forecast_df = pd.DataFrame(forecast_rescaled, index=future_dates[:len(forecast_rescaled)], columns=['Forecast'])
        
        # Display the forecast
        st.subheader("Forecasted Prices")
        st.line_chart(forecast_df)
        