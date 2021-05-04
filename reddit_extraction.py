import praw
from psaw import PushshiftAPI
import config
import datetime as dt
import numpy as np
import pandas as pd
import re
from praw.models import MoreComments
from collections import defaultdict

def get_data(dates, tickers, subreddit, n_posts=10, sort_type='created_utc'):

    # If a ticker happens in a submission, we increase its points
    title_points = config.TITLE_POINTS
    body_points = config.BODY_POINTS 
    comment_points = config.COMMENT_POINTS

    # API initialization stuff
    reddit = praw.Reddit(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        user_agent=config.USER_AGENT
    )
    api = PushshiftAPI(reddit)

    n_tickers = len(tickers)
    n_dates = len(dates)

    # Store the points for each date for each ticker
    info = pd.DataFrame(index=dates, columns=tickers)
    info.fillna(0, inplace=True)

    # Used to go back a day for the date range
    delta = dt.timedelta(days=1)

    for date_index, d in enumerate(dates):

        # Needed to parse y/m/d
        d = pd.to_datetime(d, utc=True)

        # Get the top posts from the previous day
        before = dt.datetime(
            d.year,
            d.month,
            d.day,
            0,
            0,
            tzinfo=dt.timezone.utc
        )

        # Go one day back
        after = before - delta

        # Grabs submissions between ranges
        submissions = list(api.search_submissions(
            before=int(before.timestamp()),
            after=int(after.timestamp()),
            subreddit=subreddit,
            filter=['url','author','title','subreddit','created_utc'],
            sort_type=sort_type,
            sort='desc',
            limit=n_posts
        ))


        # Go through each stock and check in each submission
        # if the stock is mentioned anywhere
        for s in submissions:
            title_text = ' '+s.title+' '
            body_text = ' '+s.selftext+' '
        
            for stock_id, stock in enumerate(tickers):
                stock_matches = [' ' + stock + ' ', ' $' + stock + ' ']
                for stock_match in stock_matches:
                    if stock_match in title_text:
                        info[stock].iloc[date_index] += title_points
                    if stock_match in body_text:
                        info[stock].iloc[date_index] += body_points
            
    return info