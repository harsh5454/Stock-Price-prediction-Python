import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(symbol, start_date, end_date):
    """
    Download stock data from Yahoo Finance
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def prepare_data(df, look_back=60, future_period=1):
    """
    Prepare data for LSTM model
    """
    # Use only closing price
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(data_scaled) - look_back - future_period + 1):
        X.append(data_scaled[i:(i + look_back)])
        y.append(data_scaled[i + look_back + future_period - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

def create_model(sequence_length):
    """
    Create LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict(symbol='AAPL', look_back=60, future_period=1):
    """
    Main function to train model and make predictions
    """
    # Get data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)
    df = get_stock_data(symbol, start_date, end_date)
    
    # Prepare data
    X, y, scaler = prepare_data(df, look_back, future_period)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model(look_back)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(y_pred_inv, label='Predicted Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Calculate and print error metrics
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    rmse = np.sqrt(mse)
    print(f'Root Mean Square Error: {rmse:.2f}')
    
    return model, scaler, history

if __name__ == "__main__":
    # Example usage
    model, scaler, history = train_and_predict('AAPL', look_back=60, future_period=1)
