import argparse

# Our libraries
from wsbot_2 import WallStreetBot as bot_2
import config

if __name__ == '__main__':
    
    # Parse arguments from command line
    # Note: Some of this is used by https://github.com/AI4Finance-LLC/FinRL, but
    # we've also added our own arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='a2c', type=str,
                        choices=['a2c', 'ddpg', 'td3', 'ppo', 'sac'],
                        help="model algorithm to use for trading")
    
    parser.add_argument('--policy_name', default='MlpPolicy', type=str,
                        choices=['MlpPolicy', 'MlpLstmPolicy'],
                        help="policy choice for training the model")
    
    parser.add_argument('--timesteps', default=1000000, type=int,
                        help="timesteps for training the RL agent")
    
    parser.add_argument('--from_saved', action='store_true',
                        help="whether to use pre-trained model. Uses start_date_test and end_date_test for trading")
    
    parser.add_argument('--saved_path', type=str,
                        help="path to saved model to load if testing")
    
    parser.add_argument('--test', default=False, action='store_true',
                        help="whether to test or train model")
    
    parser.add_argument('--tickers', default=config.DOW_30_TICKER, nargs='+',
                        help="list of stock tickers to use. default is DOW. For specific tickers, type something like: AAPL FB MSFT")
    
    parser.add_argument('--initial_balance', default=config.INITIAL_BALANCE, type=float,
                        help="balance to start training the model with")
    
    parser.add_argument('--start_date', default=config.START_DATE, type=str,
                        help="start date of training")
    parser.add_argument('--end_date', default=config.END_DATE, type=str,
                        help="end date of training")
    parser.add_argument('--start_date_test', default=config.START_DATE_TEST, type=str,
                        help="start date of testing (used with pre-trained model")
    parser.add_argument('--end_date_test', default=config.END_DATE_TEST, type=str,
                        help="end date of testing")
    
    parser.add_argument('--interval', default=config.INTERVAL, type=str,
                        choices=['1d','1h'],
                        help="trading interval or frequency to run on")
    
    parser.add_argument('--use_reddit', default=False, action='store_true',
                        help="whether to augment state space with reddit analytics")
    parser.add_argument('--reddit_path', type=str, default='',
                        help="path to reddit training data (csv dataframe)")
    parser.add_argument('--reddit_path_test', type=str, default='',
                        help="path to reddit testing data (csv dataframe)")
    parser.add_argument('--subreddit', default=config.SUBREDDIT, type=str,
                        help="subreddit name to use")
    
    parser.add_argument('--n_lstm', default=config.MlpLstmPolicy_PARAMS['n_lstm'], type=int,
                        help="size of timesteps for lstm layer (only for stable_baselines)")

    args = parser.parse_args()

    # Initialize WallStreetBot with args
    bot = bot_2(**vars(args)) 

    # If using pre-trained, then trade between test dates
    if args.from_saved:
        print(f'Trading with pre-trained model from {args.saved_path}')
        bot.trade()
    else:
        print(f'Training with model: {args.model_name} | policy: {args.policy_name} | timesteps: {args.timesteps}')
        bot.train()