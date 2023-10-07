import numpy as np

from ..noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions


class CryptoEnvContinuous(TradingEnv):

    def __init__(self, df, window_size, frame_bound, noise=False):
        assert len(frame_bound) == 2


        # Configuration Parameters
        self.frame_bound = frame_bound
        self.stop_loss = 1/100
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
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.tp_price_long = 0.0
        self.tp_price_short = 0.0
        self.sl_price_long = 0.0
        self.sl_price_short = 0.0
        self.pnl = 0.0
        self.index = 0.0
        self.tp_hit_index = None
        self.sl_hit_index = None
        self.slippage_fees = 0.0
        
        
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



    def calculate_reward_sim(self, action):

        # discourage too much skipping, (fine-tune this hyperparameter)
        if((action == Actions.Skip.value) and (self.r_active_trade == False)):
            return self.noise_function(-0.01)
        
        self.r_index += 1
        current_open = self.prices[self.last_trade_tick]
        current_low = self.hl['Low'][self.current_tick]
        current_high = self.hl['High'][self.current_tick]
        current_close = self.prices[self.current_tick]


        # ############################################################################
        # Signals creation block
        # ############################################################################
            
        self.r_long = True if (action == Actions.Buy.value) else False
        self.r_short = True if (action == Actions.Sell.value) else False
        self.r_trade_signal = (self.r_long or self.r_short) and (self.r_active_trade == False)


        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.r_trade_signal == True):
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
        # self.r_close_trade_signal = (self.r_active_trade == True) and ((self.r_current_trade_long and self.r_short) or (self.r_current_trade_short and self.r_long) or (self.r_tp_hit_index or self.r_sl_hit_index))
        self.r_close_trade_signal = (self.r_active_trade == True) and (self.r_tp_hit_index or self.r_sl_hit_index)
        
        
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

            # if TP and SL are both hit at the same candle, flip a probabilistic coin to determine the outcome
            if((self.take_profit != 0 and self.stop_loss != 0) and (self.sl_hit_index == self.tp_hit_index) and (self.sl_hit_index != None)):
                
                # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
                tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
                sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
                
                coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
                # self.coinflips.append(coinflip)
                
                # Win
                if(coinflip == 0):
                    if(self.current_trade_long and (self.high >= self.tp_price_long)):
                        self.pnl = ((((1/self.open) - (1/self.tp_price_long)) * self.current_order_size) * self.tp_price_long)
                        self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                    # maker TP short
                    if(self.current_trade_short and (self.low <= self.tp_price_short)):
                        self.pnl = ((((1/self.open) - (1/self.tp_price_short)) * (self.current_order_size * -1)) * self.tp_price_short)
                        self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                       
                # Loss
                else:
                    if(self.current_trade_long and (self.low <= self.sl_price_long)):
                        self.pnl = ((((1/self.open) - (1/self.sl_price_long)) * self.current_order_size) * self.sl_price_long)
                        self.pnl = self.pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                    # taker SL short
                    elif(self.current_trade_short and (self.high >= self.sl_price_short)):
                        self.pnl = ((((1/self.open) - (1/self.sl_price_short)) * (self.current_order_size * -1)) * self.sl_price_short)
                        self.pnl = self.pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
            
            
            # if TP is hit
            elif((self.take_profit != 0) and (self.tp_hit_index != None)):
                # maker TP long
                if(self.current_trade_long and (self.high >= self.tp_price_long)):
                    self.pnl = ((((1/self.open) - (1/self.tp_price_long)) * self.current_order_size) * self.tp_price_long)
                    self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
                # maker TP short
                if(self.current_trade_short and (self.low <= self.tp_price_short)):
                    self.pnl = ((((1/self.open) - (1/self.tp_price_short)) * (self.current_order_size * -1)) * self.tp_price_short)
                    self.pnl = self.pnl - self.current_order_size * (self.fees_maker_percentage/100)
            
            # if SL is hit
            elif((self.stop_loss != 0) and (self.r_sl_hit_index != None)):
                # taker SL long
                if(self.r_current_trade_long and (self.r_low <= self.r_sl_price_long)):
                    self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_long)) * self.current_order_size) * self.r_sl_price_long)
                    self.r_pnl = self.r_pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                # taker SL short
                elif(self.r_current_trade_short and (self.r_high >= self.r_sl_price_short)):
                    self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_short)) * (self.current_order_size * -1)) * self.r_sl_price_short)
                    self.r_pnl = self.r_pnl - (self.current_order_size * (self.fees_taker_percentage/100)) - (self.current_order_size * (self.market_fees_slippage_simulation/100))
                
            # update index variables
            self.r_sl_hit_index = None
            self.r_tp_hit_index = None
        
            # calculate the return
            self.r_pnl_return = (self.r_pnl / self.current_order_size) * 100
        
        # if a trade is currently open, give an iterim reward
        if(self.r_active_trade):
            if(self.r_current_trade_long):
                interim_reward = ((((1/self.r_open) - (1/self.r_close)) * self.current_order_size) * self.r_close)
            elif(self.r_current_trade_short):
                interim_reward = ((((1/self.r_open) - (1/self.r_close)) * (self.current_order_size * -1)) * self.r_close)
                
            # convert to return
            interim_return = (interim_reward / self.current_order_size) * 100
            return self.noise_function(interim_return / 10)
        
        return self.noise_function(self.r_pnl_return)



    def calculate_profit(self, action):
        
        self.index += 1
        current_open = self.prices[self.last_trade_tick]
        current_low = self.hl['Low'][self.current_tick]
        current_high = self.hl['High'][self.current_tick]
        current_close = self.prices[self.current_tick]
        
        
        # ############################################################################
        # Signals creation block
        # ############################################################################
            
        self.long = True if (action == Actions.Buy.value) else False
        self.short = True if (action == Actions.Sell.value) else False
        self.open_trade_signal = (self.long or self.short) and (self.active_trade == False)
        
        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.open_trade_signal == True):
            self.open = current_open
            self.active_trade = True
            self.high = current_high
            self.low = current_low
            
            self.tp_price_long = self.open + (self.open * self.take_profit/100)
            self.tp_price_short = self.open - (self.open * self.take_profit/100)
            self.sl_price_long = self.open - (self.open * self.stop_loss/100)
            self.sl_price_short = self.open + (self.open * self.stop_loss/100)
            
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
            self.close = current_close
            if(current_high > self.high):
                self.high = current_high
            if(current_low < self.low):
                self.low = current_low
            
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
        
        # self.close_trade_signal = (self.active_trade == True) and ((self.current_trade_long and self.short) or (self.current_trade_short and self.long) or (self.tp_hit_index or self.sl_hit_index))
        self.close_trade_signal = (self.active_trade == True) and (self.tp_hit_index or self.sl_hit_index)
        
        
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
                self.pnl = ((((1/self.open) - (1/self.close)) * order_size) * self.close)
            elif(self.current_trade_short):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * (Position Size * -1)) * Futures Exit Price
                self.pnl = ((((1/self.open) - (1/self.close)) * (order_size * -1)) * self.close)


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

            # if TP and SL are both hit at the same candle, flip a probabilistic coin to determine the outcome
            if((self.take_profit != 0 and self.stop_loss != 0) and (self.sl_hit_index == self.tp_hit_index) and (self.sl_hit_index != None)):
                
                # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
                tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
                sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
                
                coinflip = np.random.choice([0,1], p=[tp_prob, sl_prob])
                # self.coinflips.append(coinflip)
                
                # Win
                if(coinflip == 0):
                    if(self.current_trade_long and (self.high >= self.tp_price_long)):
                        self.pnl = ((((1/self.open) - (1/self.tp_price_long)) * order_size) * self.tp_price_long)
                        self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                    # maker TP short
                    if(self.current_trade_short and (self.low <= self.tp_price_short)):
                        self.pnl = ((((1/self.open) - (1/self.tp_price_short)) * (order_size * -1)) * self.tp_price_short)
                        self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                        
                # Loss
                else:
                    if(self.current_trade_long and (self.low <= self.sl_price_long)):
                        self.pnl = ((((1/self.open) - (1/self.sl_price_long)) * order_size) * self.sl_price_long)
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                    # taker SL short
                    elif(self.current_trade_short and (self.high >= self.sl_price_short)):
                        self.pnl = ((((1/self.open) - (1/self.sl_price_short)) * (order_size * -1)) * self.sl_price_short)
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
            
            
            # if TP is hit
            elif((self.take_profit != 0) and (self.tp_hit_index != None)):
                # maker TP long
                if(self.current_trade_long and (self.high >= self.tp_price_long)):
                    self.pnl = ((((1/self.open) - (1/self.tp_price_long)) * order_size) * self.tp_price_long)
                    self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
                # maker TP short
                if(self.current_trade_short and (self.low <= self.tp_price_short)):
                    self.pnl = ((((1/self.open) - (1/self.tp_price_short)) * (order_size * -1)) * self.tp_price_short)
                    self.pnl = self.pnl - order_size * (self.fees_maker_percentage/100)
            
            # if SL is hit
            elif((self.stop_loss != 0) and (self.sl_hit_index != None)):
                # taker SL long
                if(self.current_trade_long and (self.low <= self.sl_price_long)):
                    self.pnl = ((((1/self.open) - (1/self.sl_price_long)) * order_size) * self.sl_price_long)
                    self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                # taker SL short
                elif(self.current_trade_short and (self.high >= self.sl_price_short)):
                    self.pnl = ((((1/self.open) - (1/self.sl_price_short)) * (order_size * -1)) * self.sl_price_short)
                    self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.fees_maker_percentage/100))
                
            # update index variables
            self.sl_hit_index = None
            self.tp_hit_index = None

        return self.pnl


