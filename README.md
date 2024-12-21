# Stock-Price-prediction-Python

This repository contains a Python script that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The project utilizes historical stock price data and leverages deep learning to forecast future prices.

Features
Stock Data Download: 
>Fetches historical stock data using the Yahoo Finance API.
>Data Preprocessing: Prepares the data by scaling and creating sequences for the LSTM model.
>LSTM Model: Implements an LSTM neural network for time-series prediction.
>Visualization: Plots the actual vs. predicted stock prices for evaluation.
>Error Metrics: Computes the Root Mean Square Error (RMSE) to measure prediction accuracy.

Dependencies
Python 3.7+
numpy
pandas
yfinance
scikit-learn
tensorflow
matplotlib

Install the required packages with:
pip install -r requirements.txt

Usage 
Clone this repository:
git clone https://github.com/your-username/stock-prediction-lstm.git
cd stock-prediction-lstm

Run the script:
python stock-prediction.py

Modify parameters:
>Change the stock symbol (default: AAPL) in the train_and_predict() function.
>Adjust the look_back and future_period parameters as needed.

Script Details
Functions
>get_stock_data(symbol, start_date, end_date): Fetches stock data for the specified symbol and date range.
>prepare_data(df, look_back, future_period): Scales and transforms data for the LSTM model.
>create_model(sequence_length): Builds and compiles the LSTM model.
>train_and_predict(symbol, look_back, future_period): Orchestrates the data preparation, training, and prediction process.

Example Output
The script produces a plot comparing actual and predicted stock prices, along with the RMSE value.

Customization
>To predict other stocks, replace the symbol parameter in the train_and_predict() function.
>Tune hyperparameters such as epochs, batch_size, and look_back for optimal results.

Contributing
>Contributions are welcome! Feel free to open issues or submit pull requests for new features or bug fixes.
