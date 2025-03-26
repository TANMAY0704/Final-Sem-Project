
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def get_stock_data(ticker, start='2010-01-01'):
    data = yf.download(ticker, start=start)
    return data[['Close']]

def preprocess_lstm_data(df, column='Close', time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[column]])
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def build_stacked_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32,return_sequences=False),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    return score

def forecast_ensemble(ticker, forecast_days=30):
    # Step 1: Load Data
    df = get_stock_data(ticker)
    
    # Step 2: Preprocess
    time_step = 60
    X, y, scaler = preprocess_lstm_data(df, time_step=time_step)
    X_train, y_train = X[:-forecast_days], y[:-forecast_days]
    X_test = X[-forecast_days:]
    
    # Step 3: Train or load model
    model = build_stacked_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Step 4: Forecast using LSTM
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Step 5: Sentiment forecast
    headline = f"{ticker} stock performance"
    sentiment_score = get_sentiment_score(headline)
    sentiment_adjustment = 1 + (0.05 * sentiment_score)  # +5% if positive, -5% if negative
    adjusted_preds = predictions * sentiment_adjustment
    
    # Step 6: Plotting
    last_dates = df.index[-forecast_days:]
    plt.figure(figsize=(12, 6))
    plt.plot(last_dates, scaler.inverse_transform(y[-forecast_days:].reshape(-1, 1)), label='Actual')
    plt.plot(last_dates, predictions, label='LSTM Forecast')
    plt.plot(last_dates, adjusted_preds, label='Ensemble Forecast (with Sentiment)', linestyle='--')
    plt.title(f"{ticker} - Ensemble Forecast ({forecast_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# forecast_ensemble("TCS.NS")
