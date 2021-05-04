import config
import gym 
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
import time
# Barebones custom gym: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# Partially used: 
# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
# https://github.com/AI4Finance-LLC/FinRL

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, n_stocks, n_features, initial_balance, test=False):
        
        # Data and time constants
        self.initial_balance = initial_balance
        self.data = data # We shouldn't modify data in the initializer: assume it's already ready to go
        self.max_t = data.shape[0]
        self.test = test 
        self.max_transaction = config.MAX_TRANSACTION

        # State space: 

        # Each stock has a unique ID from 0 to n_stocks for each ticker
        # Each stock has a number of features from 0 to n_features
        # We also store the current balance of the trading agent
        # Therefore, our state space is (n_stocks*n_features) + 1

        # For each stock ID from 0 to n_stocks:
        # Feature 0 -- Amount of stocks agent holds with that ID
        # Feature 1 -- Current price of stocks with that ID
        # Feature 2:n_features -- Chosen additional features
        # Therefore, to index into the state space S for each stock ID,
        # do S[ID*n_features:(ID+1)*n_features]
        # To get the balance index, just do S[-1]
        self.n_features = n_features
        self.n_stocks = n_stocks

        self.balance_index = -1

        # Index into a stock slice with these for held and price
        self.held_index_local = 0
        self.price_index_local = 1

        self.get_held_index = lambda stock_id : (stock_id*n_features)
        self.get_price_index = lambda stock_id : (stock_id*n_features) + 1
        self.get_stock_slice = lambda stock_id : slice(stock_id*self.n_features, (stock_id+1)*self.n_features, 1)
        self.get_data_slice = lambda stock_id : slice(stock_id*(self.n_features-1), (stock_id+1)*(self.n_features-1), 1)

        self.state_dim = (self.n_stocks * self.n_features) + 1

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_dim,))

        # Action space: 
        # For each stock ID, action space ranges between -1.0 and 1.0
        # For any action a, if a < lower_hold_bound, sell |a| percentage of stocks held for that ID
        # if a > upper_hold_bound, buy |a| percentage of stocks held for that ID
        # if a between upper_hold_bound and lower_hold_bound, do nothing for that ID
        # Note that lower_hold_bound and upper_hold_bound are small fractions (like -1e-05 to 1e-05)
        # that determine the bounds for which the agent does nothing with that stock ID
        self.action_dim = (self.n_stocks)

        self.upper_hold_bound = 1e-05
        self.lower_hold_bound = -1e-05

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # Runtime variables
        self.state = None 
        self.t = 0

        # For plotting (during testing)
        self.assets_history = [self.initial_balance]
        

    def step(self, action):

        # Do one trading step
        next_t = self.t + 1

        # Initialize asset value and track asset value changes throughout process
        current_value = self.state[self.balance_index] 
        updated_value = 0.0

        for stock_id in range(self.n_stocks):
            # Refresh local balance in loop since we're actively trading now
            current_balance = self.state[self.balance_index]

            # Get all info about the stock for the current state
            stock_slice = self.get_stock_slice(stock_id)
            stock_info = self.state[stock_slice]

            # Get current price and number held for this stock
            current_held = stock_info[self.held_index_local]
            current_price = stock_info[self.price_index_local]

            # Increase our total asset value
            current_value += (current_held * current_price)

            # Parse stock action
            stock_action = action[stock_id]

            if stock_action > self.upper_hold_bound: # Buy
                
                # Clip amount bought in case we can't afford it
                amount_bought = int(stock_action*self.max_transaction)
                amount_bought_clipped = min(
                    int(current_balance//current_price),
                    amount_bought
                )

                # Update internal state with amount bought and balanced deducted
                self.state[stock_slice][self.held_index_local] += amount_bought_clipped
                self.state[self.balance_index] -= (amount_bought_clipped * current_price)

            elif stock_action < self.lower_hold_bound: # Sell
                
                # Flip polarity of action
                stock_action = np.abs(stock_action)

                # Clip amount sold in case we sell more than we own
                amount_sold = int(stock_action*self.max_transaction)
                amount_sold_clipped = min(
                    current_held,
                    amount_sold
                )

                # Update internal state with amount sold and balance gained
                self.state[stock_slice][self.held_index_local] -= amount_sold_clipped
                self.state[self.balance_index] += (amount_sold_clipped * current_price)

            else: # Hold, action is between bounds so do nothing for this stock
                pass

            # Update state with next time data
            if next_t < self.max_t:
                self.state[stock_slice][1:] = self.data[next_t, self.get_data_slice(stock_id)]

            updated_value += (self.state[stock_slice][self.held_index_local]*self.state[stock_slice][self.price_index_local])

        # Don't forget to add ending balance!
        updated_value += self.state[self.balance_index]

        
        # Advance to next time 
        self.t = next_t

        # Check if done
        if self.t >= self.max_t:
            # Our difference from the initial balance is the reward
            reward = updated_value - self.initial_balance
            done = True
            print(f'Total asset value at end of trading: {updated_value:.2f}')
        else:
            # Our asset difference before and after trading is the reward
            reward = updated_value - current_value
            done = False

        if self.test:
            self.assets_history.append(updated_value)

        return self.state, reward, done, {}

    def reset(self):
        if self.test: self.t = 0 # Always start t=0 for testing
        else: self.t = np.random.choice(self.max_t) # Choose random time for training

        # Create reset state and fill balance with initial balance
        reset_state = np.zeros(self.state_dim)
        reset_state[-1] = self.initial_balance

        # Update states for each ID from data
        for stock_id in range(self.n_stocks):
            stock_slice = self.get_stock_slice(stock_id)
            reset_state[stock_slice][1:] = self.data[self.t, self.get_data_slice(stock_id)]

        self.state = reset_state
        return self.state

    # Some from https://github.com/AI4Finance-LLC/FinRL (FinRL)
    def save_assets_history(self):
        return np.array(self.assets_history)

    def render(self, mode='human'):
        return None # Unused

    def close(self):
        pass # Unused

    # From stable_baselines documentation
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
