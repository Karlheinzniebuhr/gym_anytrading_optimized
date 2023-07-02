import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from sklearn.preprocessing import MinMaxScaler


class CryptoEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.trade_fee_percent = 0.04 / 100
        self.stop_loss = 1/100
        self.take_profit = 1.8/100
        self.order_size = 100
        
        super().__init__(df, window_size)

    ########################################################
    # Noise from https://arxiv.org/pdf/2305.02882.pdf
    # Noise ranking: https://docs.google.com/spreadsheets/d/1CTZiRX_s9RQ3sHVq6WEmWh0SonE5xDEEyDRgmVrLWEs/edit?pli=1#gid=220110982 
    ########################################################
    
    def random_uniform_scale_reward(self, step_reward):
        # Apply random uniform scale to the reward
        noise_rate = 0.01 # Probability of applying noise
        low = 0.9 # Lower boundary of the noise distribution
        high = 1.1 # Upper boundary of the noise distribution
        if np.random.rand() <= noise_rate:
            step_reward *= np.random.uniform(low, high)
        return step_reward
    
    
    # This function wasn't in the paper
    def random_normal_scale_reward(self, step_reward):
        # Apply random normal scale to the reward
        noise_rate = 1
        mean = 1
        std = 0.5
        if np.random.rand() <= noise_rate:
            step_reward *= np.random.normal(mean, std)
        return step_reward


    def random_normal_noise_reward(self, step_reward):
        # Apply random normal noise to the reward
        noise_rate = 1 # Probability of applying noise
        mean = 0 # Mean of the noise distribution
        std = 1.0 # Standard deviation of the noise distribution
        if np.random.rand() <= noise_rate:
            step_reward += np.random.normal(mean, std)
        return step_reward


    def random_uniform_noise_reward(self, step_reward):
        # Apply random uniform noise to the reward
        noise_rate = 1 # Probability of applying noise
        low = -0.001 # Lower boundary of the noise distribution
        high = 0.001 # Upper boundary of the noise distribution
        if np.random.rand() <= noise_rate:
            step_reward += np.random.uniform(low, high)
        return step_reward
    

    def process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # Get prices and scale signal features
        prices = df['Close'].to_numpy()
        hl = df[['High', 'Low']].reset_index()
        signal_features = df[['Open', 'High', 'Low', 'Close']].values
        signal_features = self.scaler.fit_transform(signal_features)

        # Compute differences and add to signal features
        diff = np.diff(prices)
        diff = np.insert(diff, 0, 0)
        signal_features = np.column_stack((signal_features, diff))

        return prices, hl, signal_features


    def calculate_reward(self, action):
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

        return self.random_normal_scale_reward(step_reward)


    def calculate_return(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self.position == Positions.Short) or
            (action == Actions.Sell.value and self.position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self.current_tick]
            last_trade_price = self.prices[self.last_trade_tick]
            price_diff = current_price - last_trade_price
            price_return = price_diff / last_trade_price

            if self.position == Positions.Short:
                step_reward += -price_return
            elif self.position == Positions.Long:
                step_reward += price_return

        return self.random_normal_scale_reward(step_reward)


    def calculate_reward_hl(self, action):
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

        return self.random_normal_scale_reward(step_reward)
    
    
    def calculate_return_hl(self, action):
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

            step_reward = step_reward / open_price

        return self.random_normal_scale_reward(step_reward)
    
    
    def calculate_reward_hl_delta(self, action):
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

        return self.random_normal_scale_reward(step_reward)

    
    def calculate_return_hl_delta(self, action):
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

            step_reward = step_reward / open_price
        return self.random_normal_scale_reward(step_reward)


    def calculate_reward_sim(self, action):
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
            coinflip = np.random.choice(np.arange(0, 2), p=[tp_prob, sl_prob])
            
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
        
        return self.random_normal_scale_reward(pnl)


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
        coinflip = np.random.choice(np.arange(0, 2), p=[tp_prob, sl_prob])
            
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
 