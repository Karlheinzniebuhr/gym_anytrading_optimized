import numpy as np

from .noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions
from sklearn.preprocessing import MinMaxScaler


class CryptoEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, training=False):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.trade_fee_percent = 0.04 / 100
        self.stop_loss = 1/100
        self.take_profit = 2/100
        self.order_size = 100
        
        # default noise function for rewards
        self.noise_function = NoiseGenerator().random_normal_scale_reward
        # uncomment this to not use any noise
        # self.noise_function = lambda x: x

        # default reward function that gets called from TradingEnv
        self.calculate_reward = self.reward_sim
        
        if(training):
            self.process_data = self.ornstein_uhlenbeck_noise
        else:
            self.process_data = self.process_data
        
        super().__init__(df, window_size)



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



    def log_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            price_diff = current_price - last_trade_price

            if self.position == Positions.Short:
                step_reward += -np.log(1 + abs(price_diff))
            elif self.position == Positions.Long:
                step_reward += np.log(1 + abs(price_diff))

        return self.noise_function(step_reward)



    def simple_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            price_diff = current_price - last_trade_price

            if self.position == Positions.Short:
                step_reward += -price_diff
            elif self.position == Positions.Long:
                step_reward += price_diff

        return self.noise_function(step_reward)



    def reward_hl(self, action):
        step_reward = 0
        
        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            open_price = self.prices[self.last_trade_tick]
            close_price = self.prices[self.current_tick]
            low = self.hl['Low'][self.current_tick]
            high = self.hl['High'][self.current_tick]
            
            if self.position == Positions.Short:
                if open_price < close_price:
                    step_reward -= abs(open_price - high)
                else:
                    step_reward += abs(open_price - low)
            elif self.position == Positions.Long:
                if open_price > close_price:
                    step_reward -= abs(open_price - low)
                else:
                    step_reward += abs(open_price - high)

            # step_reward = step_reward / open_price
        return self.noise_function(step_reward)



    def reward_hl_delta(self, action):
        step_reward = 0

        # Check if the action is a trade
        trade = ((action == Actions.Buy.value and self.position == Positions.Short) or
                (action == Actions.Sell.value and self.position == Positions.Long))

        if trade:
            # Get the prices of the current and previous ticks
            open_price = self.prices[self.last_trade_tick]
            close_price = self.prices[self.current_tick]
            low = self.hl['Low'][self.current_tick]
            high = self.hl['High'][self.current_tick]

            # Calculate the difference between the open price and the high or low price
            high_delta = abs(open_price - high)
            low_delta = abs(open_price - low)

            # Reward or penalize based on the position and the price difference
            if self.position == Positions.Short:
                step_reward = low_delta - high_delta
            elif self.position == Positions.Long:
                step_reward = high_delta - low_delta

            # step_reward = step_reward / open_price
        return self.noise_function(step_reward)



    def reward_sim(self, action):
        trade = False
        pnl = 0
        
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            low = self.hl['Low'][self.current_tick]
            high = self.hl['High'][self.current_tick]
            
            # constant order size of 1000 USD
            order_size = self.order_size
            # calculate the return as the percentage change in price
            # price_return = (current_price - last_trade_price) / last_trade_price
            sl_hit = False
            tp_hit = False

            # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
            tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
            sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
            coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
            
            if self.position == Positions.Short:
                
                stop_loss = last_trade_price + (last_trade_price * self.stop_loss)
                take_profit = last_trade_price - (last_trade_price * self.take_profit)
                
                # check if low is lower than take profit
                if(low <= take_profit):
                    tp_hit = True
                # check if high is higher than stop loss
                if(high >= stop_loss):
                    sl_hit = True
                
                # if both TP and SL are hit then we need to calculate the probability of each outcome and then randomly choose one
                if(tp_hit and sl_hit):
                
                    # Win
                    if(coinflip == 0):
                        pnl = ((((1/last_trade_price) - (1/take_profit)) * (order_size * -1)) * take_profit)
                    # Loss
                    else:
                        pnl = ((((1/last_trade_price) - (1/stop_loss)) * (order_size * -1)) * stop_loss)
                
                elif(tp_hit):
                    pnl = ((((1/last_trade_price) - (1/take_profit)) * (order_size * -1)) * take_profit)
                elif(sl_hit):
                    pnl = ((((1/last_trade_price) - (1/stop_loss)) * (order_size * -1)) * stop_loss)
                else:
                    pnl = ((((1/last_trade_price) - (1/current_price)) * (order_size * -1)) * current_price)
                
                
            elif self.position == Positions.Long:
                
                stop_loss = last_trade_price - (last_trade_price * self.stop_loss)
                take_profit = last_trade_price + (last_trade_price * self.take_profit)
                
                # check if high is higher than take profit
                if(high >= take_profit):
                    tp_hit = True
                # check if low is lower than stop loss
                if(low <= stop_loss):
                    sl_hit = True
                    
                # if both TP and SL are hit then we need to calculate the probability of each outcome and then randomly choose one    
                if(tp_hit and sl_hit):
                
                    # Win
                    if(coinflip == 0):
                        pnl = ((((1/last_trade_price) - (1/take_profit)) * order_size) * take_profit)
                    # Loss
                    else:
                        pnl = ((((1/last_trade_price) - (1/stop_loss)) * order_size) * stop_loss)
                
                elif(tp_hit):
                    pnl = ((((1/last_trade_price) - (1/take_profit)) * order_size) * take_profit)
                elif(sl_hit):
                    pnl = ((((1/last_trade_price) - (1/stop_loss)) * order_size) * stop_loss)
                else:
                    pnl = ((((1/last_trade_price) - (1/current_price)) * order_size) * current_price)
            
            
            # calculate the return
            pnl = pnl / order_size
        
        return self.noise_function(pnl)



    def calculate_profit(self, action):
        current_price = self.prices[self.current_tick]
        last_trade_price = self.prices[self.last_trade_tick]
        low = self.hl['Low'][self.current_tick]
        high = self.hl['High'][self.current_tick]
        
        # order size after fees. For simplicity the same fees are used for both maker and taker orders
        order_size = (self.total_profit * (1 - self.trade_fee_percent))
        # calculate the return as the percentage change in price
        # price_return = (current_price - last_trade_price) / last_trade_price
        pnl = 0
        sl_hit = False
        tp_hit = False

        # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
        tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
        sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
        coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
            
        trade = False
        
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade or self.done:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]

            if(self.position == Positions.Short):
                stop_loss = last_trade_price + (last_trade_price * self.stop_loss)
                take_profit = last_trade_price - (last_trade_price * self.take_profit)
                
                # check if low is lower than take profit
                if(low <= take_profit):
                    tp_hit = True
                # check if high is higher than stop loss
                if(high >= stop_loss):
                    sl_hit = True
                
                # if both TP and SL are hit then we need to calculate the probability of each outcome and then randomly choose one
                if(tp_hit and sl_hit):
                
                    # Win
                    if(coinflip == 0):
                        pnl = ((((1/last_trade_price) - (1/take_profit)) * (order_size * -1)) * take_profit)
                    # Loss
                    else:
                        pnl = ((((1/last_trade_price) - (1/stop_loss)) * (order_size * -1)) * stop_loss)
                
                elif(tp_hit):
                    pnl = ((((1/last_trade_price) - (1/take_profit)) * (order_size * -1)) * take_profit)
                elif(sl_hit):
                    pnl = ((((1/last_trade_price) - (1/stop_loss)) * (order_size * -1)) * stop_loss)
                else:
                    pnl = ((((1/last_trade_price) - (1/current_price)) * (order_size * -1)) * current_price)


            elif(self.position == Positions.Long):
                stop_loss = last_trade_price - (last_trade_price * self.stop_loss)
                take_profit = last_trade_price + (last_trade_price * self.take_profit)
                
                # check if high is higher than take profit
                if(high >= take_profit):
                    tp_hit = True
                # check if low is lower than stop loss
                if(low <= stop_loss):
                    sl_hit = True
                    
                # if both TP and SL are hit then we need to calculate the probability of each outcome and then randomly choose one    
                if(tp_hit and sl_hit):
                
                    # Win
                    if(coinflip == 0):
                        pnl = ((((1/last_trade_price) - (1/take_profit)) * order_size) * take_profit)
                    # Loss
                    else:
                        pnl = ((((1/last_trade_price) - (1/stop_loss)) * order_size) * stop_loss)
                
                elif(tp_hit):
                    pnl = ((((1/last_trade_price) - (1/take_profit)) * order_size) * take_profit)
                elif(sl_hit):
                    pnl = ((((1/last_trade_price) - (1/stop_loss)) * order_size) * stop_loss)
                else:
                    pnl = ((((1/last_trade_price) - (1/current_price)) * order_size) * current_price)
                    
                    
            # subtract exit fees.
            pnl -= (order_size + pnl) * self.trade_fee_percent
            
        return pnl
 