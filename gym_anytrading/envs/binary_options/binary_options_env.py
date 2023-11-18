import numpy as np

from ..noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions
import pandas as pd


class BinaryEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.loss_size = 1
        self.win_size = 0.95
        self.order_size = 100
        self.bet_size = 0.05
        
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
        
        self.process_data = self.process_data
        
        super().__init__(scaled_df, window_size)



    def process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # scale signal features
        signal_features = df[['High', 'Low', 'Close']].values

        hl = df[['High', 'Low']].reset_index()
        
        prices = df['Close'].to_numpy()
        diff_prices = np.insert(np.diff(prices), 0, 0)
        
        signal_features = np.column_stack((signal_features, diff_prices))

        return prices, hl, signal_features



    def reward_sim(self, action):
        pnl = 0
        n_steps_future = 0
        
        if(action == Actions.Buy.value or action == Actions.Sell.value):
            current_price = self.prices[self.current_tick + n_steps_future]
            last_trade_price = self.prices[self.current_tick - 1]
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
            pass
        
        return pnl



    def calculate_profit(self, action):
        current_price = self.prices[self.current_tick]
        last_trade_price = self.prices[self.last_trade_tick]
        # low = self.hl['Low'][self.current_tick]
        # high = self.hl['High'][self.current_tick]
        
        order_size = self.total_profit * self.bet_size
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
 