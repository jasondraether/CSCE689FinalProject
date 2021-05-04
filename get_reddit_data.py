import pandas as pd 
import stock_extraction
import reddit_extraction
import config
import datetime as dt 

# An example on how to get reddit data and save it to a csv. 
# Change subreddit, interval, start and end date and tickers however you wish

subreddit = 'wallstreetbets'
interval = '1d'
start_date = '2016-01-01'
end_date = '2020-12-31'
tickers = config.DOW_30_TICKER

stock_data = stock_extraction.get_data(
    tickers,
    interval=interval,
    start_date=start_date,
    end_date=end_date
)

reddit_data = reddit_extraction.get_data(
    stock_data['date'].unique(),
    tickers,
    subreddit,
    n_posts=50,
    sort_type='score'
)

reddit_data.to_csv(f'reddit_data_{subreddit}_{interval}_{start_date}_{end_date}.csv')