import numpy as np
import pandas as pd 
import datetime as dt

# Combine the reddit sentiment with the stock data,
# converting the reddit date to be compatible with the stock date
def add_reddit_info(stock_df, reddit_df):
    stock_df = stock_df.copy()
    reddit_df = reddit_df.copy()

    reddit_df = reddit_df.rename(columns={'Unnamed: 0':'date'})
    reddit_df['date'] = pd.to_datetime(reddit_df.date, utc=True)
    reddit_df['date'] = reddit_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))

    print(reddit_df['date'])

    tickers = stock_df['tic'].unique()
    dates = stock_df['date'].unique()

    print(stock_df['date'])

    stock_df['sentiment'] = 0

    for i, tic in enumerate(tickers):
        for j, d in enumerate(dates):
            stock_df.loc[(stock_df.tic == tic) & (stock_df.date == d)]['sentiment'] = reddit_df[tic].loc[reddit_df.date == d]

    stock_df = stock_df.sort_values(by=['date','tic'])

    return stock_df