import numpy as np
import numba as nb
import pandas as pd
import os
import pickle
import talib as ta
import sys


from ..noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions



@nb.jit
def label_dataframe(open_prices, high_prices, low_prices, close_prices, stop_loss, take_profit):
    labels_list = []  # Initialize an empty list to store label tuples

    for i in range(len(open_prices)):
        open_price = open_prices[i]
        high_price = high_prices[i]
        low_price = low_prices[i]
        close_price = close_prices[i]

        # Initialize labels for long and short
        long_label = 0
        short_label = 0

        # Assuming the trade is long, calculate take profit and stop loss prices
        take_profit_price_long = open_price * (1 + take_profit)
        stop_loss_price_long = open_price * (1 - stop_loss)

        # Assuming the trade is short, calculate take profit and stop loss prices
        take_profit_price_short = open_price * (1 - take_profit)
        stop_loss_price_short = open_price * (1 + stop_loss)

        # Initialize a counter for measuring the number of iterations
        iteration_count = 0
        
        # Track if both SL and TP are hit at the same time
        both_hit = False

        # Loop forward to check if take profit or stop loss is hit for long trade
        j = i + 1
        while j < len(open_prices):
            iteration_count += 1
            next_high = high_prices[j]
            next_low = low_prices[j]

            # Check if both stop loss and take profit are hit for long trade
            if next_high > take_profit_price_long and next_low < stop_loss_price_long:
                both_hit = True
                long_label = 0  # Label as no trade (0) if both stop loss and take profit are hit
                break

            elif next_high > take_profit_price_long:
                long_label = 1  # Label as long (1) if take profit is hit
                break
            elif next_low < stop_loss_price_long:
                long_label = -1  # Label as long (-1) if stop loss is hit
                break

            j += 1

        # Loop forward to check if take profit or stop loss is hit for short trade
        j = i + 1
        while j < len(open_prices):
            iteration_count += 1
            next_high = high_prices[j]
            next_low = low_prices[j]

            # Check if both stop loss and take profit are hit for short trade
            if next_low < take_profit_price_short and next_high > stop_loss_price_short:
                # Set both_hit to True at current index
                both_hit = True
                short_label = 0  # Label as no trade (0) if both stop loss and take profit are hit
                break

            elif next_low < take_profit_price_short:
                short_label = 1  # Label as short (1) if take profit is hit
                break
            elif next_high > stop_loss_price_short:
                short_label = -1  # Label as short (-1) if stop loss is hit
                break

            j += 1

        # Determine the final label and add it to the list as a tuple with the iteration count
        if long_label == 1:
            labels_list.append((1, iteration_count, both_hit))  # Append tuple (open_time, 1, iteration_count) if long trade is a win
        elif short_label == 1:
            labels_list.append((-1, iteration_count, both_hit))  # Append tuple (open_time, -1, iteration_count) if short trade is a win
        else:
            labels_list.append((0, iteration_count, both_hit))  # Append tuple (open_time, 0, iteration_count) if neither long nor short trade is a win

    return labels_list

    
    
class CryptoEnvContinuous(TradingEnv):

    def __init__(self, df, window_size, frame_bound, random_init_start_tick=True, noise=False):
        assert len(frame_bound) == 2

        # Configuration Parameters
        self.frame_bound = frame_bound
        self.enable_sltp = False
        self.stop_loss = 2/100
        self.take_profit = 2/100
        self.order_size = 1000
        self.current_order_size = 0.0
        self.entry_order_type = 'MARKET'
        self.exit_order_type = 'MARKET'
        self.fees_taker_percentage = 0.04
        self.fees_maker_percentage = 0.02
        self.market_fees_slippage_simulation = 0.01
        self.margin_fees = 0.0
        
        
        # Current Trade Variables Reward
        self.r_trade_signal = False
        self.r_close_trade_signal = False
        self.r_long = False
        self.r_short = False
        self.r_current_trade_long = False
        self.r_current_trade_short = False
        self.r_active_trade = False
        self.r_open = 0.0
        self.r_high = 0.0
        self.r_low = 0.0
        self.r_close = 0.0
        self.r_tp_price_long = 0.0
        self.r_tp_price_short = 0.0
        self.r_sl_price_long = 0.0
        self.r_sl_price_short = 0.0
        self.r_pnl = 0.0
        self.r_pnl_return = 0.0
        self.r_index = 0.0
        self.r_tp_hit_index = None
        self.r_sl_hit_index = None
        self.r_slippage_fees = 0.0
        
        
        # Current Trade Variables Profit
        self.trade_signal = False
        self.close_trade_signal = False
        self.long = False
        self.short = False
        self.current_trade_long = False
        self.current_trade_short = False
        self.active_trade = False
        self.candle_open = 0.0
        self.candle_high = 0.0
        self.candle_low = 0.0
        self.candle_close = 0.0
        self.tp_price_long = 0.0
        self.tp_price_short = 0.0
        self.sl_price_long = 0.0
        self.sl_price_short = 0.0
        self.pnl = 0.0
        self.index = 0.0
        self.tp_hit_index = None
        self.sl_hit_index = None
        self.slippage_fees = 0.0
        
        
        self.fixed_penalty = -0.00
        self.fixed_reward = 0.00
        self.trade_directions = []
        
        
        # default noise function for rewards
        self.noise_function = NoiseGenerator().random_normal_noise_reward
        # uncomment this to not use any noise
        # self.noise_function = lambda x: x

        # default reward function that gets called from TradingEnv
        self.calculate_reward = self.calculate_reward_sim

        if(noise):
            self.process_data = self.ornstein_uhlenbeck_noise
        else:
            self.process_data = self.process_data
            

        open_prices = np.array(df['Open'])
        high_prices = np.array(df['High'])
        low_prices = np.array(df['Low'])
        close_prices = np.array(df['Close'])
        labels_list = label_dataframe(open_prices, high_prices, low_prices, close_prices, self.stop_loss, self.take_profit)
        self.labels_df = pd.DataFrame(labels_list, columns=['label', 'iteration_count', 'both_hit'])
        
        self.debug_reward = False
        self.debug_profit = True
        
        super().__init__(df, window_size, random_init_start_tick)



    # Ornstein-Uhlenbeck noise v2
    def ornstein_uhlenbeck_noise(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :].copy()

        # Calculate the noise parameter for each row (absolute difference between high and low)
        df['noise_param'] = np.abs(df['High'] - df['Low']) + 1

        # Set theta and dt based on the noise parameter
        theta = 0.005
        df['dt'] = 0.01 / df['noise_param']

        # Generate a single random percentage between -50% and +50%
        noise_percentage = np.random.uniform(-0.5, 10)

        # Apply the noise to all rows in the DataFrame using broadcasting
        columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Create a noise array of the same shape as the DataFrame
        noise = np.full(df[columns_to_scale].shape, 1 + noise_percentage)

        # Apply the noise to the DataFrame
        df[columns_to_scale] *= noise

        # Decrease the scaling factor for less impactful noise
        scaling_factor = 0.001

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



    def process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # scale signal features
        # signal_features = df[['High', 'Low', 'Close']].values
        
        prices = df['Close'].to_numpy()
        diff_prices = np.insert(np.diff(prices), 0, 0)
        diff_high = np.insert(np.diff(df['High'].to_numpy()), 0, 0)
        diff_low = np.insert(np.diff(df['Low'].to_numpy()), 0, 0)
        
        signal_features = np.column_stack((diff_prices, diff_high, diff_low))

        hl = df[['High', 'Low']]
        date_time = df['Date']
        return date_time, prices, hl, signal_features
    
    
    
    # def process_data(self):
    #     start = self.frame_bound[0] - self.window_size
    #     end = self.frame_bound[1]
    #     df = self.df.iloc[start:end, :]
        
    #     # Calculate moving averages
    #     ma1000 = df['Close'].rolling(window=1000).mean().fillna(method='bfill').to_numpy()
    #     ma99 = df['Close'].rolling(window=99).mean().fillna(method='bfill').to_numpy()
    #     ma25 = df['Close'].rolling(window=25).mean().fillna(method='bfill').to_numpy()
    #     ma7 = df['Close'].rolling(window=7).mean().fillna(method='bfill').to_numpy()

    #     # Compute logarithmic returns
    #     log_return_ma1000 = np.log(ma1000[1:] / ma1000[:-1])
    #     log_return_ma99 = np.log(ma99[1:] / ma99[:-1])
    #     log_return_ma25 = np.log(ma25[1:] / ma25[:-1])
    #     log_return_ma7 = np.log(ma7[1:] / ma7[:-1])

    #     # Prepend zeros to maintain the shape
    #     log_return_ma1000 = np.insert(log_return_ma1000, 0, 0)
    #     log_return_ma99 = np.insert(log_return_ma99, 0, 0)
    #     log_return_ma25 = np.insert(log_return_ma25, 0, 0)
    #     log_return_ma7 = np.insert(log_return_ma7, 0, 0)

    #     # Calculate percentage changes between MAs and the current close price
    #     percentage_change_ma7_ma25 = ((ma7 - ma25) / ma25)
    #     percentage_change_ma25_ma99 = ((ma25 - ma99) / ma99)
    #     percentage_change_ma7_ma99 = ((ma7 - ma99) / ma99)
    #     percentage_change_ma7_ma1000 = ((ma7 - ma1000) / ma1000)
    #     percentage_change_ma25_ma1000 = ((ma25 - ma1000) / ma1000)
    #     percentage_change_ma99_ma1000 = ((ma99 - ma1000) / ma1000)
    #     percentage_change_close_ma7 = ((df['Close'] - ma7) / ma7)

    #     # Stack features together
    #     signal_features = np.column_stack((
    #         log_return_ma1000, log_return_ma99, log_return_ma25, log_return_ma7,
    #         percentage_change_ma7_ma25,
    #         percentage_change_ma25_ma99,
    #         percentage_change_ma7_ma99,
    #         percentage_change_ma7_ma1000,
    #         percentage_change_ma25_ma1000,
    #         percentage_change_ma99_ma1000,
    #         percentage_change_close_ma7  # Added this new feature
    #     ))

    #     prices = df['Close'].to_numpy()
    #     hl = df[['High', 'Low']].reset_index(drop=True)

    #     return prices, hl, signal_features



    # def process_data(self):
    #     start = self.frame_bound[0] - self.window_size
    #     end = self.frame_bound[1]
    #     df = self.df.iloc[start:end, :]

    #     # New FDMA with Percentage Displacement calculations
    #     length = 200
    #     displacement_percent = 0.1  # % displacement
    #     fib1 = 0.618
    #     fib2 = 0.382
    #     fib3 = 0.236
    #     log_scale = False  # Set according to your needs

    #     src = np.log(df['Close']) if log_scale else df['Close']
    #     ma = src.rolling(window=length).mean()

    #     # Calculate displacement as a percentage of the MA
    #     displacement = ma * displacement_percent
    #     ma_fib1_disp = displacement * fib1
    #     ma_fib2_disp = displacement * fib2
    #     ma_fib3_disp = displacement * fib3

    #     ma_fib1_upper = ma + ma_fib1_disp
    #     ma_fib1_lower = ma - ma_fib1_disp
    #     ma_fib2_upper = ma + ma_fib2_disp
    #     ma_fib2_lower = ma - ma_fib2_disp
    #     ma_fib3_upper = ma + ma_fib3_disp
    #     ma_fib3_lower = ma - ma_fib3_disp

    #     # Calculate normalized distances for MAs and bands
    #     normalized_distance_close_ma = (df['Close'] - ma) / ma
    #     normalized_distance_close_ma_fib1_upper = (df['Close'] - ma_fib1_upper) / ma_fib1_upper
    #     normalized_distance_close_ma_fib1_lower = (df['Close'] - ma_fib1_lower) / ma_fib1_lower
    #     normalized_distance_close_ma_fib2_upper = (df['Close'] - ma_fib2_upper) / ma_fib2_upper
    #     normalized_distance_close_ma_fib2_lower = (df['Close'] - ma_fib2_lower) / ma_fib2_lower
    #     normalized_distance_close_ma_fib3_upper = (df['Close'] - ma_fib3_upper) / ma_fib3_upper
    #     normalized_distance_close_ma_fib3_lower = (df['Close'] - ma_fib3_lower) / ma_fib3_lower

    #     # Stack new features together with existing signal features
    #     signal_features = np.column_stack((normalized_distance_close_ma,
    #                                     normalized_distance_close_ma_fib1_upper,
    #                                     normalized_distance_close_ma_fib1_lower,
    #                                     normalized_distance_close_ma_fib2_upper,
    #                                     normalized_distance_close_ma_fib2_lower,
    #                                     normalized_distance_close_ma_fib3_upper,
    #                                     normalized_distance_close_ma_fib3_lower))

    #     prices = df['Close'].to_numpy()
    #     hl = df[['High', 'Low']].reset_index(drop=True)

    #     return prices, hl, signal_features



    def calculate_reward_sim_multiverse(self, action):
        # Lookup if the current action is a win or loss
        mv_label = self.labels_df.iloc[self.current_tick]
        win_loss_skip_label = mv_label['label']
        iteration_count = mv_label['iteration_count']

        # Normalize the reward 
        reward_weight = (1.0 + (1.0 / iteration_count)) * 2
        
        if action == Actions.Skip.value:
            # Skip condition
            if win_loss_skip_label == 0:
                # Skipped correctly
                return 0.0
            elif (win_loss_skip_label == 1) or (win_loss_skip_label == -1):
                return 0.0
                # Opportunity cost
                # loss_reward = (-self.stop_loss * self.order_size)
                # return loss_reward

        elif (action == Actions.Buy.value and win_loss_skip_label == 1) or (action == Actions.Sell.value and win_loss_skip_label == -1):
            # Win condition: Action is in favor (Buy and label is 1) or (Sell and label is -1)
            # Calculate the win reward proportionally to take_profit
            win_reward = reward_weight * self.take_profit * self.order_size
            return win_reward

        elif (action == Actions.Buy.value and win_loss_skip_label == -1) or (action == Actions.Sell.value and win_loss_skip_label == 1):
            # Loss condition: Action is against (Buy and label is -1) or (Sell and label is 1)
            # Calculate the loss reward proportionally to stop_loss
            loss_reward = -self.stop_loss * self.order_size
            return loss_reward

        else:
            # Loss condition: Action is against (Buy and label is 0) or (Sell and label is 0)
            # Calculate the loss reward proportionally to stop_loss
            loss_reward = (-self.stop_loss * self.order_size)
            return loss_reward



    def calculate_reward_sim(self, action):
        
        self.r_pnl = 0.0
        
        # Lookup if the current action is a win or loss
        # win_loss_skip_label = self.labels_df.iloc[self.last_trade_tick]['label']
        
        if((action == Actions.Skip.value) and (self.r_active_trade == False)):
            # interim_reward = self.calculate_reward_sim_multiverse(action)
            if(self.debug_reward):
                print(f'Skip at open: {self.prices[self.current_tick - 1]}, reward: {self.fixed_penalty}')
            return self.fixed_penalty
        
        
        self.r_index += 1
        current_open = self.prices[self.current_tick - 1]
        current_low = self.hl['Low'].iloc[self.current_tick]
        current_high = self.hl['High'].iloc[self.current_tick]
        current_close = self.prices[self.current_tick]


        # ############################################################################
        # Signals creation block
        # ############################################################################
        
        self.r_long = True if (action == Actions.Buy.value) else False
        self.r_short = True if (action == Actions.Sell.value) else False
        self.skip = True if (action == Actions.Skip.value) else False
        self.r_trade_signal = (self.r_long or self.r_short) and (self.r_active_trade == False)


        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.r_trade_signal == True):
            
            # debug code
            if(self.debug_reward):
                if(self.r_long):
                    print(f'Long at open: {current_open}')
                elif (self.r_short):
                    print(f'Short at open: {current_open}')
            
            
            self.r_open = current_open
            self.r_active_trade = True
            self.r_high = current_high
            self.r_low = current_low
            
            self.r_tp_price_long = self.r_open + (self.r_open * self.take_profit)
            self.r_tp_price_short = self.r_open - (self.r_open * self.take_profit)
            self.r_sl_price_long = self.r_open - (self.r_open * self.stop_loss)
            self.r_sl_price_short = self.r_open + (self.r_open * self.stop_loss)
            
            if(self.r_long):
                self.r_current_trade_long = True
                self.r_current_trade_short = False
            elif(self.r_short):
                self.r_current_trade_short = True
                self.r_current_trade_long = False
                
            self.trade_directions.append(action)
                
            # ############################################################################
            # Order Size. 
            # ############################################################################
            
            self.current_order_size = self.order_size


        # ############################################################################
        # Update high/low, close at each iteration of the open trade. 
        # Track if SL or TP are hit first
        # ############################################################################
        
        if(self.r_active_trade):
            self.r_close = current_close
            if(current_high > self.r_high):
                self.r_high = current_high
            if(current_low < self.r_low):
                self.r_low = current_low
            
            # Check if SL or TP are hit, assign current index to self.r_tp_hit_index or self.r_sl_hit_index
            if(self.enable_sltp):
                if(self.r_long and self.r_sl_hit_index == None):
                    if(current_low <= self.r_sl_price_long):
                        self.r_sl_hit_index = self.r_index
                    if(current_high >= self.r_tp_price_long):
                        self.r_tp_hit_index = self.r_index
                if(self.r_short and self.r_sl_hit_index == None):
                    if(current_high >= self.r_sl_price_short):
                        self.r_sl_hit_index = self.r_index
                    if(current_low <= self.r_tp_price_short):
                        self.r_tp_hit_index = self.r_index        
                    
                    
        # ########################################################
        # Close trade when signal condition is met
        # If open_trade=True, close trade. Then open next trade
        # ########################################################
        
        if(self.enable_sltp):
            self.r_close_trade_signal = (
                (self.r_active_trade == True) and
                (
                    (self.r_tp_hit_index != None) or
                    (self.r_sl_hit_index != None) or
                    (self.r_current_trade_long and self.r_short) or 
                    (self.r_current_trade_short and self.r_long) or
                    (self.skip)
                )
            )
        else:
            self.r_close_trade_signal = (
                (self.r_active_trade == True) and
                (
                    (self.r_current_trade_long and self.r_short) or 
                    (self.r_current_trade_short and self.r_long) or
                    (self.skip)
                )
            )


        # ########################################################
        # Close trade when signal condition is met
        # If open_trade=True, close trade. Then open next trade
        # ########################################################
            
        if(self.r_active_trade and self.r_close_trade_signal):
            
            # reset variables
            self.r_trade_signal = False
            self.r_close_trade_signal = False
            self.r_active_trade = False


            # ############################################################################
            # Open-Close PNL. 
            # ############################################################################
            
            # Basic PnL
            if(self.r_current_trade_long):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * Position Size) * Futures Exit Price
                self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * self.current_order_size) * self.r_close)
            elif(self.r_current_trade_short):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * (Position Size * -1)) * Futures Exit Price
                self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * (self.current_order_size * -1)) * self.r_close)


            # Trading Fee = (Open Value X Maker Fee Rate) + (Close Value X Maker Fee Rate)
            if(self.entry_order_type == 'MARKET'):
                self.entry_fees = self.current_order_size * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.r_slippage_fees = self.current_order_size * (self.market_fees_slippage_simulation/100)
            if(self.entry_order_type == 'LIMIT'):
                self.entry_fees = self.current_order_size * (self.fees_maker_percentage/100)

            if(self.exit_order_type == 'MARKET'):
                self.exit_fees = (self.current_order_size + self.r_pnl) * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.r_slippage_fees = self.current_order_size * (self.market_fees_slippage_simulation/100)
            if(self.exit_order_type == 'LIMIT'):
                self.exit_fees = (self.current_order_size + self.r_pnl) * (self.fees_maker_percentage/100)

            self.margin_fees = self.current_order_size * (self.margin_fees/100)

            self.current_trade_total_fees = self.entry_fees + self.exit_fees + self.r_slippage_fees + self.margin_fees
            self.r_pnl = self.r_pnl - self.current_trade_total_fees


            # ##############################################
            # TP/SL & liquidation PNL
            # ##############################################

            if(self.enable_sltp):
                # if TP and SL are both hit at the same candle, flip a probabilistic coin to determine the outcome
                if((self.sl_hit_index == self.tp_hit_index) and (self.sl_hit_index != None)):
                    
                    # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
                    tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
                    sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
                    
                    coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
                    # self.coinflips.append(coinflip)
                    
                    # Win
                    if(coinflip == 0):
                        if(self.current_trade_long and (self.candle_high >= self.tp_price_long)):
                            self.pnl = ((((1/self.candle_open) - (1/self.tp_price_long)) * self.current_order_size) * self.tp_price_long)
                            self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                        # maker TP short
                        if(self.current_trade_short and (self.candle_low <= self.tp_price_short)):
                            self.pnl = ((((1/self.candle_open) - (1/self.tp_price_short)) * (self.current_order_size * -1)) * self.tp_price_short)
                            self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                        
                    # Loss
                    else:
                        if(self.current_trade_long and (self.candle_low <= self.sl_price_long)):
                            self.pnl = ((((1/self.candle_open) - (1/self.sl_price_long)) * self.current_order_size) * self.sl_price_long)
                            self.pnl = self.pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                        # taker SL short
                        elif(self.current_trade_short and (self.candle_high >= self.sl_price_short)):
                            self.pnl = ((((1/self.candle_open) - (1/self.sl_price_short)) * (self.current_order_size * -1)) * self.sl_price_short)
                            self.pnl = self.pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                
                
                # if TP is hit
                if(self.tp_hit_index != None):
                    # maker TP long
                    if(self.current_trade_long and (self.candle_high >= self.tp_price_long)):
                        self.pnl = ((((1/self.candle_open) - (1/self.tp_price_long)) * self.current_order_size) * self.tp_price_long)
                        self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                    # maker TP short
                    if(self.current_trade_short and (self.candle_low <= self.tp_price_short)):
                        self.pnl = ((((1/self.candle_open) - (1/self.tp_price_short)) * (self.current_order_size * -1)) * self.tp_price_short)
                        self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                
                # if SL is hit
                elif(self.r_sl_hit_index != None):
                    # taker SL long
                    if(self.r_current_trade_long and (self.r_low <= self.r_sl_price_long)):
                        self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_long)) * self.current_order_size) * self.r_sl_price_long)
                        self.r_pnl = self.r_pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                    # taker SL short
                    elif(self.r_current_trade_short and (self.r_high >= self.r_sl_price_short)):
                        self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_short)) * (self.current_order_size * -1)) * self.r_sl_price_short)
                        self.r_pnl = self.r_pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                    
                
            # ##############################################
            # Reset variables
            # ##############################################
            self.r_sl_hit_index = None
            self.r_tp_hit_index = None
            self.r_current_trade_long = False
            self.r_current_trade_short = False
            self.position = Positions.Long
            self.current_pnl = 0.
            
            # calculate the % return
            # self.r_pnl_return = (self.r_pnl / self.current_order_size) * 100
        
        # if a trade is currently open, give an iterim reward
        if(self.r_active_trade):
            # if(self.r_current_trade_long):
            #     # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * Position Size) * Futures Exit Price
            #     self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * self.current_order_size) * self.r_close)
            # elif(self.r_current_trade_short):
            #     # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * (Position Size * -1)) * Futures Exit Price
            #     self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * (self.current_order_size * -1)) * self.r_close)
                
            # self.current_pnl = (self.r_pnl / self.current_order_size) * 100
             
            if(self.debug_reward):
                print(f'fixed reward at close {current_close}: {self.fixed_reward}')
            return self.fixed_reward #interim_reward_return
        
            
        if(self.debug_reward):
            print(f'reward at close {current_close}: {(self.r_pnl / self.current_order_size) * 100}')
            
        return (self.r_pnl / self.current_order_size) * 100



    def calculate_profit(self, action):
        
        self.pnl = 0
        
        self.index += 1
        current_open = self.prices[self.current_tick - 1]
        current_low = self.hl['Low'].iloc[self.current_tick]
        current_high = self.hl['High'].iloc[self.current_tick]
        current_close = self.prices[self.current_tick]
        
        
        # ############################################################################
        # Signals creation block
        # ############################################################################
            
        self.long = True if (action == Actions.Buy.value) else False
        self.short = True if (action == Actions.Sell.value) else False
        self.skip = True if (action == Actions.Skip.value) else False
        self.open_trade_signal = (self.long or self.short) and (self.active_trade == False)
        
        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.open_trade_signal == True):
            
            # debug code
            if(self.debug_profit):
                if(self.r_long):
                    print(f'Long at open: {current_open}, date_time: {self.date_time.iloc[self.current_tick]}')
                elif (self.r_short):
                    print(f'Short at open: {current_open} date_time: {self.date_time.iloc[self.current_tick]}')
                    
            self.candle_open = current_open
            self.active_trade = True
            self.candle_high = current_high
            self.candle_low = current_low
            
            self.tp_price_long = self.candle_open + (self.candle_open * self.take_profit)
            self.tp_price_short = self.candle_open - (self.candle_open * self.take_profit)
            self.sl_price_long = self.candle_open - (self.candle_open * self.stop_loss)
            self.sl_price_short = self.candle_open + (self.candle_open * self.stop_loss)
            
            if(self.long):
                self.current_trade_long = True
                self.current_trade_short = False
            elif(self.short):
                self.current_trade_short = True
                self.current_trade_long = False


        # ############################################################################
        # Update high/low, close at each iteration of the open trade. 
        # Track if SL or TP are hit first
        # ############################################################################
        
        if(self.active_trade):
            self.candle_close = current_close
            if(current_high > self.candle_high):
                self.candle_high = current_high
            if(current_low < self.candle_low):
                self.candle_low = current_low
            
            if(self.enable_sltp):
                # Check if SL or TP are hit, assign current index to self.tp_hit_index or self.sl_hit_index
                if(self.long and self.sl_hit_index == None):
                    if(current_low <= self.sl_price_long):
                        self.sl_hit_index = self.index
                    if(current_high >= self.tp_price_long):
                        self.tp_hit_index = self.index
                if(self.short and self.sl_hit_index == None):
                    if(current_high >= self.sl_price_short):
                        self.sl_hit_index = self.index
                    if(current_low <= self.tp_price_short):
                        self.tp_hit_index = self.index        
                    

        # ########################################################
        # Close trade when signal condition is met
        # If open_trade=True, close trade. Then open next trade
        # ########################################################
        
        # if(self.current_tick == self.end_tick):
        #     print(f'End of backtest.')
        
        if(self.enable_sltp):
            self.close_trade_signal = (
                (self.active_trade == True) and
                (
                    (self.tp_hit_index != None) or
                    (self.sl_hit_index != None) or
                    (self.current_trade_long and self.short) or
                    (self.current_trade_short and self.long) or
                    (self.skip) or
                    (self.current_tick == self.end_tick)
                )
            )
        else:
            self.close_trade_signal = (
                (self.active_trade == True) and
                (
                    (self.current_trade_long and self.short) or
                    (self.current_trade_short and self.long) or
                    (self.skip) or
                    (self.current_tick == self.end_tick)
                )
            )
            
        
        # ########################################################
        # Close trade when signal condition is met
        # ########################################################
            
        if(self.active_trade and self.close_trade_signal):
            
            # reset variables
            self.open_trade_signal = False
            self.close_trade_signal = False
            self.active_trade = False

            # ############################################################################
            # Open-Close PNL. 
            # ############################################################################
                
            # use entire balance
            order_size = self.total_profit
            
            # Basic PnL
            if(self.current_trade_long):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * Position Size) * Futures Exit Price
                self.pnl = ((((1/self.candle_open) - (1/self.candle_close)) * order_size) * self.candle_close)
            elif(self.current_trade_short):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * (Position Size * -1)) * Futures Exit Price
                self.pnl = ((((1/self.candle_open) - (1/self.candle_close)) * (order_size * -1)) * self.candle_close)


            # Trading Fee = (Open Value X Maker Fee Rate) + (Close Value X Maker Fee Rate)
            if(self.entry_order_type == 'MARKET'):
                self.entry_fees = order_size * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.slippage_fees = order_size * (self.market_fees_slippage_simulation/100)
            if(self.entry_order_type == 'LIMIT'):
                self.entry_fees = order_size * (self.fees_maker_percentage/100)

            if(self.exit_order_type == 'MARKET'):
                self.exit_fees = (order_size + self.pnl) * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.slippage_fees = order_size * (self.market_fees_slippage_simulation/100)
            if(self.exit_order_type == 'LIMIT'):
                self.exit_fees = (order_size + self.pnl) * (self.fees_maker_percentage/100)

            self.margin_fees = order_size * (self.margin_fees/100)

            self.current_trade_total_fees = self.entry_fees + self.exit_fees + self.slippage_fees + self.margin_fees
            self.pnl = self.pnl - self.current_trade_total_fees


            # ##############################################
            # TP/SL & liquidation PNL
            # ##############################################

            if(self.enable_sltp):
                # if TP and SL are both hit at the same candle, flip a probabilistic coin to determine the outcome
                if((self.sl_hit_index == self.tp_hit_index) and (self.sl_hit_index != None)):
                    
                    # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
                    tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
                    sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
                    
                    coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
                    # self.coinflips.append(coinflip)
                    
                    # Win
                    if(coinflip == 0):
                        if(self.current_trade_long and (self.candle_high >= self.tp_price_long)):
                            self.pnl = ((((1/self.candle_open) - (1/self.tp_price_long)) * order_size) * self.tp_price_long)
                            self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                        # maker TP short
                        if(self.current_trade_short and (self.candle_low <= self.tp_price_short)):
                            self.pnl = ((((1/self.candle_open) - (1/self.tp_price_short)) * (order_size * -1)) * self.tp_price_short)
                            self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                            
                    # Loss
                    else:
                        if(self.current_trade_long and (self.candle_low <= self.sl_price_long)):
                            self.pnl = ((((1/self.candle_open) - (1/self.sl_price_long)) * order_size) * self.sl_price_long)
                            self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                        # taker SL short
                        elif(self.current_trade_short and (self.candle_high >= self.sl_price_short)):
                            self.pnl = ((((1/self.candle_open) - (1/self.sl_price_short)) * (order_size * -1)) * self.sl_price_short)
                            self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                
                
                # if TP is hit
                if(self.tp_hit_index != None):
                    # maker TP long
                    if(self.current_trade_long and (self.candle_high >= self.tp_price_long)):
                        self.pnl = ((((1/self.candle_open) - (1/self.tp_price_long)) * order_size) * self.tp_price_long)
                        self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                    # maker TP short
                    if(self.current_trade_short and (self.candle_low <= self.tp_price_short)):
                        self.pnl = ((((1/self.candle_open) - (1/self.tp_price_short)) * (order_size * -1)) * self.tp_price_short)
                        self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                
                # if SL is hit
                elif(self.sl_hit_index != None):
                    # taker SL long
                    if(self.current_trade_long and (self.candle_low <= self.sl_price_long)):
                        self.pnl = ((((1/self.candle_open) - (1/self.sl_price_long)) * order_size) * self.sl_price_long)
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                    # taker SL short
                    elif(self.current_trade_short and (self.candle_high >= self.sl_price_short)):
                        self.pnl = ((((1/self.candle_open) - (1/self.sl_price_short)) * (order_size * -1)) * self.sl_price_short)
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                    
                
            # ##############################################
            # Reset variables
            # ##############################################
            self.sl_hit_index = None
            self.tp_hit_index = None
            self.current_trade_long = False
            self.current_trade_short = False

        if(self.debug_profit):
            print(f'profit at close {current_close}: {self.pnl}, date_time: {self.date_time.iloc[self.current_tick]}')
        return self.pnl

