import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from sklearn.preprocessing import MinMaxScaler


class CryptoEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.scaler = MinMaxScaler()
        self.trade_fee_percent = 0.04
        self.stop_loss = 1/100
        self.take_profit = 2/100
        self.order_size = 1000
        
        super().__init__(df, window_size)


    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # Get prices and scale signal features
        prices = df['Close'].to_numpy()
        hl = df[['High', 'Low']].reset_index()
        scaler = MinMaxScaler()
        signal_features = df[['Open', 'High', 'Low', 'Close']].values
        signal_features = scaler.fit_transform(signal_features)

        # Compute differences and add to signal features
        diff = np.diff(prices)
        diff = np.insert(diff, 0, 0)
        signal_features = np.column_stack((signal_features, diff))

        return prices, hl, signal_features


    def _calculate_reward(self, action):
        pnl = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            low = self.hl['Low'][self._current_tick]
            high = self.hl['High'][self._current_tick]
            
            # constant order size of 1000 USD
            order_size = self.order_size
            # calculate the return as the percentage change in price
            # price_return = (current_price - last_trade_price) / last_trade_price
            pnl = 0
            sl_hit = False
            tp_hit = False

            # calculate the probability of each outcome based on the TP/SL distance. Example if TP is 0.5% and SL is 0.1% then the probability of TP being hit is 0.833 and SL being hit is 0.167
            tp_prob = self.stop_loss / (self.take_profit + self.stop_loss)
            sl_prob = self.take_profit / (self.take_profit + self.stop_loss)
            coinflip = np.random.choice(np.arange(0, 2), p=[tp_prob, sl_prob])
            
            if self._position == Positions.Short:
                
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
                
                
            elif self._position == Positions.Long:
                
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
                
        return pnl


    def calculate_profit(self, action):
        trade = False
        total_profit = 0
        
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            shares = (self._total_profit * (1 - self.trade_fee_percent)) / last_trade_price
            
            if(self._position == Positions.Long):
                total_profit = (shares * (1 - self.trade_fee_percent)) * (current_price - last_trade_price)
            elif(self._position == Positions.Short):
                total_profit = (shares * (1 - self.trade_fee_percent)) * (last_trade_price - current_price)
                
        return total_profit
 

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = self._total_profit

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            shares = (profit * (1 - self.trade_fee_percent)) / last_trade_price
            
            if(position == Positions.Long):
                profit += (shares * (1 - self.trade_fee_percent)) * (current_price - last_trade_price)
            elif(position == Positions.Short):
                profit += (shares * (1 - self.trade_fee_percent)) * (last_trade_price - current_price)

            last_trade_tick = current_tick - 1
            
        return profit