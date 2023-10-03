import numpy as np

from ..noise import NoiseGenerator
from .trading_env import TradingEnv, Actions, Positions


class CryptoEnvContinuous(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2


        # Configuration Parameters
        self.frame_bound = frame_bound
        self.stop_loss = 1/100
        self.take_profit = 2/100
        self.order_size = 100
        self.entry_order_type = 'MARKET'
        self.exit_order_type = 'MARKET'
        self.fees_taker_percentage = 0.04
        self.fees_maker_percentage = 0.02
        self.market_fees_slippage_simulation = 0.01
        self.margin_fees = 0
        
        
        # Current Trade Variables Reward
        self.r_trade_signal = False
        self.r_close_trade_signal = False
        self.r_current_trade_long = False
        self.r_current_trade_short = False
        self.r_active_trade = False
        self.r_open = 0
        self.r_high = 0
        self.r_low = 0
        self.r_close = 0
        self.r_tp_price_long = 0
        self.r_tp_price_short = 0
        self.r_sl_price_long = 0
        self.r_sl_price_short = 0
        self.r_pnl = 0
        self.r_pnl_return = 0
        self.r_index = 0
        self.r_tp_hit_index = None
        self.r_sl_hit_index = None
        self.r_slippage_fees = 0
        
        # Current Trade Variables Profit
        self.trade_signal = False
        self.close_trade_signal = False
        self.current_trade_long = False
        self.current_trade_short = False
        self.active_trade = False
        self.open = 0
        self.high = 0
        self.low = 0
        self.close = 0
        self.tp_price_long = 0
        self.tp_price_short = 0
        self.sl_price_long = 0
        self.sl_price_short = 0
        self.pnl = 0
        self.index = 0
        self.tp_hit_index = None
        self.sl_hit_index = None
        self.slippage_fees = 0
        
        
        # default noise function for rewards
        self.noise_function = NoiseGenerator().random_normal_scale_reward
        # uncomment this to not use any noise
        # self.noise_function = lambda x: x

        # default reward function that gets called from TradingEnv
        self.calculate_reward = self.calculate_reward_sim
        
        super().__init__(df, window_size)

    
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
            return -0.0001
        
        self.r_index += 1
        self.current_open = self.prices[self.last_trade_tick]
        self.current_low = self.hl['Low'][self.current_tick]
        self.current_high = self.hl['High'][self.current_tick]
        self.current_close = self.prices[self.current_tick]
        
        
        # ############################################################################
        # Signals creation block
        # ############################################################################
            
        self.long = True if Actions.Buy.value else False
        self.short = True if Actions.Sell.value else False
        self.r_current_trade_short = (self.position == Positions.Short)
        self.r_current_trade_long = (self.position == Positions.Long)
        self.r_trade_signal = (self.long or self.short) and (self.r_active_trade == False)
        

        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.r_trade_signal == True):
            self.r_open = self.current_open
            self.r_active_trade = True
            self.r_high = self.current_high
            self.r_low = self.current_low
            
            self.r_tp_price_long = self.r_open + (self.r_open * self.take_profit/100)
            self.r_tp_price_short = self.r_open - (self.r_open * self.take_profit/100)
            self.r_sl_price_long = self.r_open - (self.r_open * self.stop_loss/100)
            self.r_sl_price_short = self.r_open + (self.r_open * self.stop_loss/100)
            
            if(self.long):
                self.r_current_trade_long = True
                self.r_current_trade_short = False
            elif(self.short):
                self.r_current_trade_short = True
                self.r_current_trade_long = False


        # ############################################################################
        # Update high/low, close at each iteration of the open trade. 
        # Track if SL or TP are hit first
        # ############################################################################
        
        if(self.r_active_trade):
            self.r_close = self.current_close
            if(self.current_high > self.r_high):
                self.r_high = self.current_high
            if(self.current_low < self.r_low):
                self.r_low = self.current_low
            
            # Check if SL or TP are hit, assign current index to self.r_tp_hit_index or self.r_sl_hit_index
            if(self.long and self.r_sl_hit_index == None):
                if(self.current_low <= self.r_sl_price_long):
                    self.r_sl_hit_index = self.r_index
                if(self.current_high >= self.r_tp_price_long):
                    self.r_tp_hit_index = self.r_index
            if(self.short and self.r_sl_hit_index == None):
                if(self.current_high >= self.r_sl_price_short):
                    self.r_sl_hit_index = self.r_index
                if(self.current_low <= self.r_tp_price_short):
                    self.r_tp_hit_index = self.r_index        
                    
                    
        # ########################################################
        # Close trade when signal condition is met
        # If open_trade=True, close trade. Then open next trade
        # ########################################################
        # self.r_close_trade_signal = (self.r_active_trade == True) and ((self.r_current_trade_long and self.short) or (self.r_current_trade_short and self.long) or (self.r_tp_hit_index or self.r_sl_hit_index))
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
            # Order Size. 
            # ############################################################################
            
            order_size = self.order_size


            # ############################################################################
            # Open-Close PNL. 
            # ############################################################################
            
            # Basic PnL
            if(self.r_current_trade_long):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * Position Size) * Futures Exit Price
                self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * order_size) * self.r_close)
            elif(self.r_current_trade_short):
                # (((1 / Futures Entry Price) - (1 / Futures Exit Price)) * (Position Size * -1)) * Futures Exit Price
                self.r_pnl = ((((1/self.r_open) - (1/self.r_close)) * (order_size * -1)) * self.r_close)


            # Trading Fee = (Open Value X Maker Fee Rate) + (Close Value X Maker Fee Rate)
            if(self.entry_order_type == 'MARKET'):
                self.entry_fees = order_size * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.r_slippage_fees += order_size * (self.market_fees_slippage_simulation/100)
            if(self.entry_order_type == 'LIMIT'):
                self.entry_fees = order_size * (self.fees_maker_percentage/100)

            if(self.exit_order_type == 'MARKET'):
                self.exit_fees = (order_size + self.r_pnl) * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.r_slippage_fees += order_size * (self.market_fees_slippage_simulation/100)
            if(self.exit_order_type == 'LIMIT'):
                self.exit_fees = (order_size + self.r_pnl) * (self.fees_maker_percentage/100)

            self.margin_fees = order_size * (self.margin_fees/100)

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
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.market_fees_slippage_simulation/100))
                    # taker SL short
                    elif(self.current_trade_short and (self.high >= self.sl_price_short)):
                        self.pnl = ((((1/self.open) - (1/self.sl_price_short)) * (order_size * -1)) * self.sl_price_short)
                        self.pnl = self.pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.market_fees_slippage_simulation/100))
            
            
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
            elif((self.stop_loss != 0) and (self.r_sl_hit_index != None)):
                # taker SL long
                if(self.r_current_trade_long and (self.r_low <= self.r_sl_price_long)):
                    self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_long)) * order_size) * self.r_sl_price_long)
                    self.r_pnl = self.r_pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.market_fees_slippage_simulation/100))
                # taker SL short
                elif(self.r_current_trade_short and (self.r_high >= self.r_sl_price_short)):
                    self.r_pnl = ((((1/self.r_open) - (1/self.r_sl_price_short)) * (order_size * -1)) * self.r_sl_price_short)
                    self.r_pnl = self.r_pnl - (order_size * (self.fees_taker_percentage/100)) - (order_size * (self.market_fees_slippage_simulation/100))
                
            # update index variables
            self.r_sl_hit_index = None
            self.r_tp_hit_index = None
        
            # calculate the return
            self.r_pnl_return = self.r_pnl / order_size
        
        # if a trade is open, give an iterim reward
        # if(self.r_active_trade):
        #     return 0.001
        
        return self.r_pnl_return


    def calculate_profit(self, action):
        
        self.index += 1
        self.current_open = self.prices[self.last_trade_tick]
        self.current_low = self.hl['Low'][self.current_tick]
        self.current_high = self.hl['High'][self.current_tick]
        self.current_close = self.prices[self.current_tick]
        
        
        # ############################################################################
        # Signals creation block
        # ############################################################################
            
        self.long = True if Actions.Buy.value else False
        self.short = True if Actions.Sell.value else False
        self.current_trade_short = (self.position == Positions.Short)
        self.current_trade_long = (self.position == Positions.Long)
        self.open_trade_signal = (self.long or self.short) and (self.active_trade == False)
        self.close_trade_signal = (self.active_trade == True) and ((self.current_trade_long and self.short) or (self.current_trade_short and self.long) or (self.tp_hit_index or self.sl_hit_index))
        
        # ########################################################
        # If conditions are OK, open trade and set open variables.
        # ########################################################
        
        if(self.open_trade_signal == True):
            self.open = self.current_open
            self.active_trade = True
            self.high = self.current_high
            self.low = self.current_low
            
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
            self.close = self.current_close
            if(self.current_high > self.high):
                self.high = self.current_high
            if(self.current_low < self.low):
                self.low = self.current_low
            
            # Check if SL or TP are hit, assign current index to self.tp_hit_index or self.sl_hit_index
            if(self.long and self.sl_hit_index == None):
                if(self.current_low <= self.sl_price_long):
                    self.sl_hit_index = self.index
                if(self.current_high >= self.tp_price_long):
                    self.tp_hit_index = self.index
            if(self.short and self.sl_hit_index == None):
                if(self.current_high >= self.sl_price_short):
                    self.sl_hit_index = self.index
                if(self.current_low <= self.tp_price_short):
                    self.tp_hit_index = self.index        
                    
                    
        # ########################################################
        # Close trade when signal condition is met
        # If open_trade=True, close trade. Then open next trade
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
            order_size = self.order_size
            
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
                    self.slippage_fees += order_size * (self.market_fees_slippage_simulation/100)
            if(self.entry_order_type == 'LIMIT'):
                self.entry_fees = order_size * (self.fees_maker_percentage/100)

            if(self.exit_order_type == 'MARKET'):
                self.exit_fees = (order_size + self.pnl) * (self.fees_taker_percentage/100)
                if(self.market_fees_slippage_simulation):
                    self.slippage_fees += order_size * (self.market_fees_slippage_simulation/100)
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


