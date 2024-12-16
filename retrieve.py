import os
import sys
# import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from rinfo import API_KEY

def fetch_stock_data(ticker, api_key, start_date, end_date):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')

    data = data[(data.index >= start_date) & (data.index <= end_date)]

    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Date'}, inplace=True)

    return data

def save_to_csv(data, ticker):
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{ticker}.csv')
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1]
    api_key = API_KEY
    start_date = '2010-01-01'
    end_date = '2024-10-03'

    data = fetch_stock_data(ticker, api_key, start_date, end_date)
    save_to_csv(data, ticker)
