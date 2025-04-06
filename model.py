# Enhanced Model Script for Indian Stock Sentiment Analysis

import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import date
from nsepy import get_history
import os

# Load FinBERT for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model_sentiment = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')


def get_sentiment_score(headline):
    inputs = tokenizer(headline, return_tensors='pt', truncation=True)
    outputs = model_sentiment(**inputs)
    probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
    neg, neu, pos = probs
    return pos - neg


def fetch_news_and_get_sentiment(ticker, start, end):
    # Placeholder: Implement actual news fetching logic
    dates = pd.date_range(start, end)
    sentiment_scores = np.random.uniform(-1, 1, len(dates))  # Simulated scores
    return pd.DataFrame({'Date': dates, 'Sentiment': sentiment_scores})


def create_training_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Example usage: Train on TCS data
ticker = 'TCS.NS'
df = yf.download(ticker, period='5y', auto_adjust=True)
sentiment_df = fetch_news_and_get_sentiment(ticker, df.index[0], df.index[-1])
sentiment_df.set_index('Date', inplace=True)
df = df.join(sentiment_df, how='left').fillna(0)
data = df[['Close', 'Sentiment']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = create_training_sequences(data_scaled)
X = X.reshape((X.shape[0], X.shape[1], data_scaled.shape[1]))

model = build_model((60, 2))
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
model.save('TCS_model.h5')
