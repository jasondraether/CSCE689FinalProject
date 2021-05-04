# CSCE689FinalProject
Final project for CSCE 689: Reinforcement Learning

config.py --  Configuration file. Dates, tickers, reddit API settings are stored here, along with any constants needed for the code.

get_reddit_data.py -- An example on extracting reddit data from any subreddit based on dates grabbed from stock extraction. Will save data as a csv.

main.py -- Runs the program. Command line arguments covered at the bottom.

models.py -- Essentially a wrapper class to train and test with models from the stable_baselines library.

plot.py -- Used for plotting total assets overtime.

reddit_data_1d.csv -- Sample reddit sentiment data for 1d interval from WallStreetBets, collected between 2016-01-01 to 2020-12-31.

reddit_data_test_1d.csv -- Sample reddit sentiment data for 1d interval from WallStreetBets, collected between 2021-01-01 to 2021-04-30.

reddit_extraction.py -- Used to pull data from reddit using PRAW and Pushshift. Does so by using dates from stock extraction as before and after ranges. Also does the sentiment analysis to calculate each tickers sentiment score.

stock_extraction.py -- Pulls Yahoo! Finance data on stock tickers using the yfinance API. Also calculates Moving Average Convergence Divergence (MACD).

trading_env.py -- Custom OpenAI Gym environment created for stock trading.

utils.py -- Used to combine the Reddit sentiment data with the stock data.

wsbot_2.py -- WallStreetBot class. Called from main.py to essentially do the entire training and testing process.

Example on running A2C with 1,000,000 timesteps with MLP policy WITH reddit sentiment and an initial balance of 10000 dollars:

python3 main.py --model_name a2c --policy_name MlpPolicy --initial_balance 10000 --use_reddit --reddit_path reddit_data_1d.csv --reddit_path_test reddit_data_test_1d.csv --start_date 2016-01-01 --end_date 2020-12-31 --start_date_test 2021-01-01 --end_date_test 2021-04-30 --timesteps 1000000

Example on running A2C with the same configuration WITHOUT reddit sentiment:

python3 main.py --model_name a2c --policy_name MlpPolicy --initial_balance 10000 --timesteps 1000000
