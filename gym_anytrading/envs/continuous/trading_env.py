import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import pandas as pd



class Actions(Enum):
    Sell = 0
    Buy = 1
    Skip = 2


class Positions(Enum):
    Short = 0
    Long = 1
    NoPosition = 2


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, random_init_start_tick=True):
        # Assert that the dataframe has the correct dimensions
        assert df.ndim == 2

        # Initialize random number generator
        self.seed()

        # Dataframe and window settings
        self.df = df
        self.window_size = window_size
        self.max_ma_size = 200  # Maximum size for moving averages

        # Processing input data
        self.date_time, self.ohlc, self.signal_features = self.process_data()

        # Action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.shape = (window_size, self.signal_features.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # Episode control variables
        self.start_tick = self.window_size + self.max_ma_size
        self.end_tick = len(self.ohlc) - 1
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.done = False
        self.truncated = False
        self.position = Positions.NoPosition
        self.first_rendering = True

        # Reward and profit tracking
        self.total_reward = 0.
        self.total_profit = 1000.
        self.current_pnl = 0.

        # Debugging and history tracking
        self.debug_rewards = []
        self.debug_profits = []
        self.history = None

        # Environment specification and initialization settings
        self.spec = gym.envs.registration.EnvSpec('ContinuousEnv-v0')
        self.random_init_start_tick = random_init_start_tick

        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def reset(self, seed=None, options=None):
        # Reset the environment state
        super().reset(seed=seed, options=options)

        # Decide start tick based on random initialization setting
        if self.random_init_start_tick:
            # Sample random start tick between window_size and end_tick
            self.start_tick = random.randint(self.window_size + self.max_ma_size, self.end_tick - self.window_size)
        else:
            self.start_tick = self.window_size + self.max_ma_size

        # Reset episode control variables
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.done = False
        self.truncated = False
        self.position = Positions.NoPosition
        self.first_rendering = True

        # Reset reward, profit, and history tracking
        self.total_reward = 0.
        self.total_profit = 1000.
        self.current_pnl = 0.
        self.history = {}

        # Optional: Re-generate new observations
        # self.date_time, self.ohlc, self.signal_features = self.process_data()

        # Return initial observation and history
        return self.get_observation(), self.history



    def step(self, action):
        
        if self.current_tick >= self.end_tick:
            self.truncated = True
            self.done = True
        else:
            self.current_tick += 1
            
        # If reward reaches zero, end the episode
        # if(self.total_reward <= 0):
        #     self.truncated = True
            
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        
        self.debug_rewards.append(step_reward)
        if np.isnan(self.debug_rewards).any():
            print(self.debug_rewards)
            assert False
        
        profit, trade_opened_or_closed = self.calculate_profit(action)
        self.total_profit += profit
        
        self.debug_profits.append(profit)
        # if(self.total_profit <= 100):
        #     print("Total profit is less than 100")

        # if trade was made, update position
        if(action == Actions.Buy.value):
            self.position = Positions.Long
            # update the last trade tick
            self.last_trade_tick = self.current_tick
        
        elif(action == Actions.Sell.value):
            self.position = Positions.Short
            # update the last trade tick
            self.last_trade_tick = self.current_tick
            
        # if action was skip, no position is taken
        elif(action == Actions.Skip.value):
            self.position = Positions.NoPosition
        
        info = dict(
            date_time = self.date_time.iloc[self.current_tick],
            open = self.ohlc.iloc[self.current_tick]['Open'],
            high = self.ohlc.iloc[self.current_tick]['High'],
            low = self.ohlc.iloc[self.current_tick]['Low'],
            close = self.ohlc.iloc[self.current_tick]['Close'],
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            position = self.position.value,
            trade_opened_or_closed = trade_opened_or_closed,
        )
        self.update_history(info)

        observation = self.get_observation()
        
        return observation, step_reward, self.done, self.truncated, info


    def get_observation(self):
        # return self.signal_features[(self.current_tick - self.window_size + 1):self.current_tick + 1]
        start_tick = self.current_tick - self.window_size + 1
        end_tick = self.current_tick + 1
        # return np.append(np.append(self.signal_features[start_tick : end_tick].flatten(), self.position.value), self.current_pnl)
        # return np.append(self.signal_features[start_tick : end_tick].flatten(), self.position.value)
        return self.signal_features[start_tick : end_tick]


    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.ohlc['Close'][tick], color=color)

        if self.first_rendering:
            self.first_rendering = False
            plt.cla()
            plt.plot(self.ohlc['Close'])
            start_position = self.history['position'][self.start_tick]
            plot_position(start_position, self.start_tick)

        plot_position(self.position, self.current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit
        )

        plt.pause(0.01)


        
    def render_all_positions(self, mode='human'):
        if not self.history or 'position' not in self.history or 'date_time' not in self.history:
            print("Required data (position and date_time) is not available in self.history")
            return

        position_history = self.history['position']
        date_time_history = self.history['date_time']
        trade_history = self.history['trade_opened_or_closed']
        open_history = self.history['open']
        high_history = self.history['high']
        low_history = self.history['low']
        close_history = self.history['close']
        window_ticks = np.arange(len(position_history))
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=window_ticks,
                                    open=open_history,
                                    high=high_history,
                                    low=low_history,
                                    close=close_history,
                                    name='Candlesticks'))

        # Accumulate position markers and count positions
        short_x, short_y, short_text = [], [], []
        long_x, long_y, long_text = [], [], []
        num_short = 0
        num_long = 0
        num_no_position = 0
        
        open_trade_index = None
        open_trade_position = None
        
        for i, (position, trade_status) in enumerate(zip(position_history, trade_history)):
            date_time = date_time_history[i] if i < len(date_time_history) else "Unknown"
            if position == Positions.Short.value:
                short_x.append(window_ticks[i])
                short_y.append(open_history[i])
                short_text.append(date_time)
                num_short += 1
            elif position == Positions.Long.value:
                long_x.append(window_ticks[i])
                long_y.append(open_history[i])
                long_text.append(date_time)
                num_long += 1
            elif position == Positions.NoPosition.value:
                num_no_position += 1
                
            # Check trade status and draw trade lines
            if trade_status == 1:  # Trade opened
                open_trade_index = window_ticks[i]
                open_trade_position = position
            elif trade_status == 2 and open_trade_index is not None:  # Trade closed
                close_trade_index = window_ticks[i]
                
                
                # Determine the color based on the position
                line_color = "green" if open_trade_position == Positions.Long.value else "red"

                fig.add_shape(type="line",
                              x0=open_trade_index, y0=open_history[open_trade_index], 
                              x1=close_trade_index, y1=open_history[close_trade_index],
                              line=dict(color=line_color, width=2))
                open_trade_index = None


        # Add position marker traces with hover text
        fig.add_trace(go.Scatter(x=short_x, y=short_y, mode='markers', marker=dict(color='red', size=10), name='Short',
                                text=short_text, hoverinfo='text+x+y'))
        fig.add_trace(go.Scatter(x=long_x, y=long_y, mode='markers', marker=dict(color='green', size=10), name='Long',
                                text=long_text, hoverinfo='text+x+y'))

        # Annotations
        annotations = [
            dict(
                xref='paper', yref='paper',
                x=0.95, y=0.95,
                text=f'Short Positions: {num_short}<br>' +
                    f'Long Positions: {num_long}<br>' +
                    f'No Position: {num_no_position}',
                showarrow=False,
                align='right',
                font=dict(family='Courier New, monospace', size=12),
                bgcolor='rgba(255, 255, 255, 0.5)',
                xanchor='right',
                yanchor='top'
            )
        ]
        fig.update_layout(annotations=annotations)

        # Update layout
        fig.update_layout(
            title='Continuous Test Environment<br>Total Reward: %.6f ~ Total Profit: %.6f' % (self.total_reward, self.total_profit),
            xaxis_title='Ticks',
            yaxis_title='Price',
            height=800
        )

        fig.update_layout(xaxis_rangeslider_visible=True, dragmode='zoom', yaxis=dict(fixedrange=False))
        
        fig.show()



    def plot_reward(self):
        # Create a Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.history['total_reward'],
                                mode='lines',
                                name='total_reward'))
        fig.update_layout(title='Total Reward')

        # Set Plotly to dark mode
        fig.update_layout(template='plotly_dark')

        # Show the figure
        fig.show()



    def plot_profit(self, log=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.history['total_profit'],
                                mode='lines',
                                name='total_profit'))
        fig.update_layout(title='Total Profit')

        # Set Plotly to dark mode
        fig.update_layout(template='plotly_dark')

        if(log):
            fig.update_yaxes(type="log")

        fig.show()

        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def process_data(self):
        raise NotImplementedError


    def calculate_reward(self, action):
        raise NotImplementedError


    def calculate_profit(self, action):
        raise NotImplementedError
