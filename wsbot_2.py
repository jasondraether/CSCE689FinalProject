import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

# Our libraries
import stock_extraction
import reddit_extraction
from trading_env import TradingEnv
import config
import utils
from models import DRLAgent

class WallStreetBot(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.output_model_dir = f"./trained_models/{self.model_name}_{self.policy_name}/start_balance_{self.initial_balance}_nlstm_{self.n_lstm}"
        self.output_result_dir = f"./results/{self.model_name}_{self.policy_name}/start_balance_{self.initial_balance}_nlstm_{self.n_lstm}"

        # Make directory if it doesn't exist
        if not os.path.exists(self.output_model_dir): os.makedirs(self.output_model_dir)
        if not os.path.exists(self.output_result_dir): os.makedirs(self.output_result_dir)

    def train(self):
        # Some from https://github.com/AI4Finance-LLC/FinRL (FinRL)
        # but we've adapted to suit our needs 
        # The general programatic flow is mostly used

        print("======== Get data ========")
        # use our environment
        df_data_train = stock_extraction.get_data(
            self.tickers,
            interval = self.interval,
            start_date = self.start_date,
            end_date = self.end_date
        )

        df_data_test = stock_extraction.get_data(
            self.tickers,
            interval = self.interval,
            start_date = self.start_date_test,
            end_date = self.end_date_test
        )

        # Get all unique dates in dataframe
        train_dates = df_data_train['date'].unique().tolist()
        test_dates = df_data_test['date'].unique().tolist()

        # Append reddit data if we're using reddit
        if self.use_reddit:
            if self.reddit_path != '' and self.reddit_path_test != '':
                reddit_data = pd.read_csv(self.reddit_path)
                reddit_data_test = pd.read_csv(self.reddit_path_test)
            else:
                reddit_data = reddit_extraction.get_data(
                    train_dates,
                    self.tickers,
                    self.subreddit
                )
                reddit_data_test = reddit_extraction.get_data(
                    test_dates,
                    self.tickers,
                    self.subreddit
                )

            df_data_train = utils.add_reddit_info(df_data_train, reddit_data)
            df_data_test = utils.add_reddit_info(df_data_test, reddit_data_test)

        # Extract and flatten the features we're using for training and testing
        train_info = []
        for t in train_dates:
            date_info = df_data_train[df_data_train['date'] == t]
            if self.use_reddit: train_info.append(date_info[['open','macd','sentiment']].values.flatten())
            else: train_info.append(date_info[['open','macd']].values.flatten())
        train_info = np.array(train_info)

        test_info = []
        for t in test_dates:
            date_info = df_data_test[df_data_test['date'] == t]
            if self.use_reddit: test_info.append(date_info[['open','macd','sentiment']].values.flatten())
            else: test_info.append(date_info[['open','macd']].values.flatten())
        test_info = np.array(test_info)

        # If we use reddit, need to make sure to initialize environment with 4 features, not 3
        if self.use_reddit:
            # 4 features
            env_train, _ = TradingEnv(train_info, len(self.tickers), 4, self.initial_balance).get_sb_env()
            env_test = TradingEnv(test_info, len(self.tickers), 4,  self.initial_balance, test=True)
        else:
            # 3 features
            env_train, _ = TradingEnv(train_info, len(self.tickers), 3,  self.initial_balance).get_sb_env()
            env_test = TradingEnv(test_info, len(self.tickers), 3,  self.initial_balance, test=True)
        
        # Initialize agent, get model, and train model
        agent = DRLAgent(env=env_train,
                    save_dir=self.output_model_dir)

        print("=========================================")
        print(f"Train the agent from {self.start_date} to {self.end_date}")
        print("=========================================")
        Model = agent.get_model(model_name=self.model_name, policy=self.policy_name)
        
        trained_model = agent.train_model(
            model=Model, total_timesteps=self.timesteps, save_path=self.output_model_dir
        )
        print("============= Training done =============")
        print("=========================================")
        print(f"Trading from {self.start_date_test} to {self.end_date_test}")
        print("=========================================")

        # Initialize test agent, using trained model get predictions on test data
        assets_history = DRLAgent.DRL_prediction(
            model=trained_model, environment=env_test)

        print("Initial balance: {}, Final balance: {}".format(
                    assets_history[0], assets_history[-1]))
        
        np.savetxt(
            f"{self.output_result_dir}/{self.initial_balance}_{self.model_name}_{self.policy_name}_{self.timesteps}_{self.interval}.csv",
            assets_history,
            delimiter=','
        )

    def trade(self):

        # Using pre-trained model, extract testing data and get predictions 
        # (basically the same as train, just without any training loops)
        df_data_test = stock_extraction.get_data(
            self.tickers,
            interval = self.interval,
            start_date = self.start_date_test,
            end_date = self.end_date_test
        )

        # Get all unique dates in dataframe
        test_dates = df_data_test['date'].unique().tolist()

        # Append reddit data if we're using reddit
        if self.use_reddit:
            if self.reddit_path_test != '':
                reddit_data_test = pd.read_csv(self.reddit_path_test)
            else:
                reddit_data_test = reddit_extraction.get_data(
                    test_dates,
                    self.tickers,
                    self.subreddit
                )

            df_data_test = utils.add_reddit_info(df_data_test, reddit_data_test)

        test_info = []
        for t in test_dates:
            date_info = df_data_test[df_data_test['date'] == t]
            if self.use_reddit: test_info.append(date_info[['open','macd','sentiment']].values.flatten())
            else: test_info.append(date_info[['open','macd']].values.flatten())
        test_info = np.array(test_info)

        if self.use_reddit:
            # 4 features
            env_test = TradingEnv(test_info, len(self.tickers), 4,  self.initial_balance, test=True)
        else:
            # 3 features
            env_test = TradingEnv(test_info, len(self.tickers), 3,  self.initial_balance, test=True)
        
        agent = DRLAgent(env=env_test)

        model = agent.load_model(self.model_name, self.saved_path)

        assets_history = DRLAgent.DRL_prediction(
            model=model, environment=env_test)

        print("Initial balance: {}, Final balance: {}".format(
                    assets_history[0], assets_history[-1]))
        
        np.savetxt(
            f"{self.output_result_dir}/trade_{self.initial_balance}_{self.model_name}_{self.policy_name}_{self.timesteps}_{self.interval}.csv",
            assets_history,
            delimiter=','
        )