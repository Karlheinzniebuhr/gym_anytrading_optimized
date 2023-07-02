import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from sklearn.preprocessing import MinMaxScaler


class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left'):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        self.trade_fee_percent = 0.0003  # unit
        self.scaler = MinMaxScaler()
        
        super().__init__(df, window_size)


    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        df = self.df.iloc[start:end, :]

        # Get prices and scale signal features
        prices = df['Close'].to_numpy()
        scaler = MinMaxScaler()
        signal_features = df[['Open', 'High', 'Low', 'Close']].values
        signal_features = scaler.fit_transform(signal_features)

        # Compute differences and add to signal features
        diff = np.diff(prices)
        diff = np.insert(diff, 0, 0)
        signal_features = np.column_stack((signal_features, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_return = current_price - last_trade_price
            
            if self._position == Positions.Short:
                step_reward += -price_return * 10000
            elif self._position == Positions.Long:
                step_reward += price_return * 10000

        return step_reward


    def calculate_profit(self, action):
        trade = False
        total_profit = 0
        
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            quantity = (self._total_profit * (1 - self.trade_fee_percent)) / last_trade_price

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    total_profit = (last_trade_price - current_price) * quantity
                elif self._position == Positions.Long:
                    total_profit = (current_price - last_trade_price) * quantity

            elif self.unit_side == 'right':
                if self._position == Positions.Short:
                    total_profit = (current_price - last_trade_price) * quantity / current_price
                elif self._position == Positions.Long:
                    total_profit = (last_trade_price - current_price) * quantity / current_price

        return total_profit


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

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

            quantity = (self._total_profit * (1 - self.trade_fee_percent)) / last_trade_price

            if self.unit_side == 'left':
                if position == Positions.Short:
                    profit = 1 - (last_trade_price - current_price) * quantity
                elif position == Positions.Long:
                    profit = 1 + (current_price - last_trade_price) * quantity

            elif self.unit_side == 'right':
                if position == Positions.Short:
                    profit = 1 - (current_price - last_trade_price) * quantity / current_price
                elif position == Positions.Long:
                    profit = 1 + (last_trade_price - current_price) * quantity / current_price

            last_trade_tick = current_tick - 1

        return profit


