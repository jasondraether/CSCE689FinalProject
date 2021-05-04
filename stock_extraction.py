import pandas as pd 
import numpy as np 
import yfinance as yf
from stockstats import StockDataFrame as Sdf
import datetime as dt

def get_stock_data(tickers, interval, period=None, start_date='', end_date=''):
    # Taken from https://github.com/AI4Finance-LLC/FinRL (FinRL)
    data_df = pd.DataFrame()
    for tic in tickers:
        # Check if period is defined
        if period != None:
            ticker_df = yf.download(tic, period=period, interval=interval)
        else:
            ticker_df = yf.download(tic, start=start_date, end=end_date, interval=interval)

        ticker_df['tic'] = tic
        data_df = data_df.append(ticker_df)

    # Renames columns to place nice with stockstats
    data_df = data_df.reset_index()

    data_df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjcp",
        "volume",
        "tic",
    ]
    # Replace close with adjusted close, then drop adjusted close
    data_df['close'] = data_df['adjcp']
    data_df = data_df.drop('adjcp', axis='columns')

    data_df['date'] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))

    # Instead of dropping, fill using interpolation methods
    data_df = data_df.fillna(method="bfill").fillna(method="ffill")

    # Add MACD
    data_df = add_macd(data_df)

    # Keep only the entries we need
    data_df = data_df[['date','tic','open','macd']]
    data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

    return data_df

# Mostly from https://github.com/AI4Finance-LLC/FinRL (FinRL)
def add_macd(df):
    
    # Basically extract MACD using stockstats and combines it 
    # with our original stock data as a new column
    df = df.copy()
    df = df.sort_values(by=['tic','date'])
    
    stock = Sdf.retype(df.copy())
    
    unique_ticker = stock.tic.unique()
    
    macd_df = pd.DataFrame()
    indicator = 'macd'
    for i in range(len(unique_ticker)):
        try:
            temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
            temp_indicator = pd.DataFrame(temp_indicator)
            temp_indicator['tic'] = unique_ticker[i]
            temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
            macd_df = macd_df.append(
                temp_indicator, ignore_index=True
            )
        except Exception as e:
           print(e)

    df = df.merge(macd_df[['tic','date',indicator]],on=['tic','date'],how='left')
    df = df.sort_values(by=['tic','date'])

    return df


def get_data(tickers, interval, period=None, start_date='', end_date=''):

    df_all = get_stock_data(tickers, interval, period, start_date, end_date)

    # Fix any missing entries
    df_all = df_all.fillna(method='bfill').fillna(method='ffill') 

    return df_all