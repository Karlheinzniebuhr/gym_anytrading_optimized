import numpy as np

from ..noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class BinaryEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, training=False):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.trade_fee_percent = 0.04 / 100
        self.loss_size = 1
        self.win_size = 0.85
        self.order_size = 100
        self.bet_size = 0.01
        
        # default noise function for rewards
        self.noise_function = NoiseGenerator().random_normal_scale_reward
        # uncomment this to not use any noise
        # self.noise_function = lambda x: x

        # default reward function that gets called from TradingEnv
        self.calculate_reward = self.reward_sim
        
        # Scale df
        scaled_df = df.copy()
        max_decimal_places = df['Close'].apply(lambda x: abs(x % 1)).max()
        scaling_factor = 10 ** len(str(max_decimal_places).split('.')[1])
        scaled_df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']] * scaling_factor
        
        if(training):
            self.process_data = self.ornstein_uhlenbeck_noise
        else:
            self.process_data = self.process_data
        
        super().__init__(scaled_df, window_size)



    def process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # scale signal features
        signal_features = df[['Open', 'High', 'Low', 'Close']].values

        hl = df[['High', 'Low']].reset_index()
        
        prices = df['Close'].to_numpy()
        diff_prices = np.insert(np.diff(prices), 0, 0)
        
        signal_features = np.column_stack((signal_features, diff_prices))

        return prices, hl, signal_features



    ##################################################################
    # OBSERVATION NOISE
    ##################################################################

    # Ornstein-Uhlenbeck noise v2
    def ornstein_uhlenbeck_noise(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :].copy()

        # Calculate the noise parameter for each row (absolute difference between high and low)
        df['noise_param'] = np.abs(df['High'] - df['Low'])

        # Set theta and dt based on the noise parameter
        theta = 0.005
        df['dt'] = np.where(df['noise_param'] == 0, 0, 0.01 / df['noise_param'])

        # Decrease the scaling factor for less impactful noise
        scaling_factor = 0.01

        # Initialize a single noise term for each row
        noise_per_row = np.zeros(df.shape[0])
        # Vectorized calculation of noise_per_row
        mu_values = df['noise_param'].values
        dt_values = df['dt'].values
        noise_per_row[1:] = np.cumsum(
            theta * (mu_values[1:] - noise_per_row[:-1]) * dt_values[1:] +
            np.sqrt(dt_values[1:]) * np.random.normal(size=df.shape[0] - 1)
        )

        # Vectorized operation to add noise to df_noisy_ou
        df[['Open', 'High', 'Low', 'Close']] += scaling_factor * noise_per_row[:, np.newaxis]

        # Continue with the rest of your data processing steps
        signal_features = df[['Open', 'High', 'Low', 'Close']].values
        hl = df[['High', 'Low']].reset_index()

        prices = df['Close'].to_numpy()
        diff_prices = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((signal_features, diff_prices))

        return prices, hl, signal_features
    
    # Normal noise
    def normal_noise(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # Define parameters for normal noise
        noise_mean = 1  # The mean of the normal distribution
        noise_std = 0.01  # The standard deviation of the normal distribution

        # Generate normal noise with the given standard deviation
        noise_open = np.random.normal(noise_mean, noise_std, size=df.shape[0])
        noise_high = np.random.normal(noise_mean, noise_std, size=df.shape[0])
        noise_low = np.random.normal(noise_mean, noise_std, size=df.shape[0])
        noise_close = np.random.normal(noise_mean, noise_std, size=df.shape[0])

        # Create a new DataFrame df_noisy to store noisy data
        df_noisy = df.copy()

        # Apply normal noise to each column of df_noisy
        df_noisy['Open'] *= noise_open
        df_noisy['High'] *= noise_high
        df_noisy['Low'] *= noise_low
        df_noisy['Close'] *= noise_close

        # Scale signal features
        signal_features = df_noisy[['Open', 'High', 'Low', 'Close']].values

        hl = df_noisy[['High', 'Low']].reset_index()

        # Calculate the differences in prices
        prices = df_noisy['Close'].to_numpy()  # Extract Close prices
        diff_prices = np.insert(np.diff(prices), 0, 0)

        # Add the differences in prices to the signal features
        signal_features = np.column_stack((signal_features, diff_prices))

        return prices, hl, signal_features
    
    

    def reward_sim(self, action):
        pnl = 0
        
        if(action == Actions.Buy.value or action == Actions.Sell.value):
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            # low = self.hl['Low'][self.current_tick]
            # high = self.hl['High'][self.current_tick]
            
            # if the close is higher than the open and the action was buy then we have a win
            if(current_price > last_trade_price and action == Actions.Buy.value):
                pnl = self.order_size * self.win_size * self.bet_size
            # if the close is lower than the open and the action was sell then we have a win
            elif(current_price < last_trade_price and action == Actions.Sell.value):
                pnl = self.order_size * self.win_size * self.bet_size
            # if the close is lower than the open and the action was buy then we have a loss
            elif(current_price < last_trade_price and action == Actions.Buy.value):
                pnl =- self.order_size * self.loss_size * self.bet_size
            # if the close is higher than the open and the action was sell then we have a loss
            elif(current_price > last_trade_price and action == Actions.Sell.value):
                pnl =- self.order_size * self.loss_size * self.bet_size
            
        # if action = Skip
        else:
            pnl =- (self.order_size * self.loss_size * self.bet_size)/10
        
        return self.noise_function(pnl)



    def calculate_profit(self, action):
        current_price = self.prices[self.current_tick]
        last_trade_price = self.prices[self.last_trade_tick]
        # low = self.hl['Low'][self.current_tick]
        # high = self.hl['High'][self.current_tick]
        
        order_size = self.total_profit
        pnl = 0
        
        # if bet_size is bigger than 0 then we have a trade
        if(action == Actions.Buy.value or action == Actions.Sell.value):

            # if the close is higher than the open and the action was buy then we have a win
            if(current_price > last_trade_price and action == Actions.Buy.value):
                pnl = order_size * self.win_size * self.bet_size
            # if the close is lower than the open and the action was sell then we have a win
            elif(current_price < last_trade_price and action == Actions.Sell.value):
                pnl = order_size * self.win_size * self.bet_size
            # if the close is lower than the open and the action was buy then we have a loss
            elif(current_price < last_trade_price and action == Actions.Buy.value):
                pnl =- order_size * self.loss_size * self.bet_size
            # if the close is higher than the open and the action was sell then we have a loss
            elif(current_price > last_trade_price and action == Actions.Sell.value):
                pnl =- order_size * self.loss_size * self.bet_size
                
        
        else:
            pass
            
        return pnl
 