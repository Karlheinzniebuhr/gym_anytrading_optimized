import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1
    Skip = 2


class Positions(Enum):
    Short = 0
    Long = 1
    NoPosition = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.hl, self.signal_features = self.process_data()

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.shape = (window_size, self.signal_features.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self.start_tick = self.window_size
        self.end_tick = len(self.prices) - 1
        self.done = False
        self.truncated = False
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.position = Positions.NoPosition
        self.position_history = (self.window_size * [None]) + [self.position]
        self.total_reward = 0.
        self.total_profit = 1000.
        self.first_rendering = True
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.done = False
        self.truncated = False
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.position = Positions.NoPosition
        self.position_history = (self.window_size * [None]) + [self.position]
        self.total_reward = 0.
        self.total_profit = 1000.
        self.first_rendering = True
        self.history = {}
        return self.get_observation(), self.history


    def step(self, action):
        
        self.done = False
        self.truncated = False
        self.current_tick += 1

        if self.current_tick == self.end_tick:
            self.done = True

        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        
        profit = self.calculate_profit(action)
        self.total_profit += profit

        # if trade was made, update position
        if(action == Actions.Buy.value):
            self.position = Positions.Long
        elif(action == Actions.Sell.value):
            self.position = Positions.Short
            
        # if action was skip, no position is taken
        elif(action == Actions.Skip.value):
            self.position = Positions.NoPosition
        
        self.last_trade_tick = self.current_tick
            
        self.position_history.append(self.position)
        observation = self.get_observation()
        info = dict(
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            position = self.position.value
        )
        self.update_history(info)

        return observation, step_reward, self.done, self.truncated, info


    def get_observation(self):
        return self.signal_features[(self.current_tick - self.window_size + 1):self.current_tick + 1]


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
                plt.scatter(tick, self.prices[tick], color=color)

        if self.first_rendering:
            self.first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self.position_history[self.start_tick]
            plot_position(start_position, self.start_tick)

        plot_position(self.position, self.current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self.position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self.position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self.position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit + ' ~ '
        )
        
        
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
