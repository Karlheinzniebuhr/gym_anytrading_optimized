import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import random


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
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.hl, self.signal_features = self.process_data()
        
        # Added this for MA features
        self.max_ma_size = 0

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        # self.shape = (window_size, self.signal_features.shape[1])
        self.shape = (window_size * self.signal_features.shape[1] + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self.start_tick = self.window_size + self.max_ma_size
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
        
        self.debug_rewards = []
        # used for observation PnL
        self.current_pnl = 0.
        
        # Set the environment specification id
        self.spec = gym.envs.registration.EnvSpec('ContinuousEnv-v0')
        
        # Set to true to randomly initialize the environment
        self.random_init_start_tick = random_init_start_tick
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.done = False
        self.truncated = False
        
        if(self.random_init_start_tick):
            # Sample random start tick between window_size and end_tick
            self.start_tick = random.randint(self.window_size, self.end_tick - self.window_size)
        
        else:
            self.start_tick = self.window_size + self.max_ma_size
        
        # Adjust current tick and last trade tick accordingly
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        
        self.position = Positions.NoPosition
        self.position_history = (self.window_size * [None]) + [self.position]
        self.total_reward = 0.
        self.total_profit = 1000.
        self.first_rendering = True
        self.history = {}
        
        # generate new noisy observations
        # self.prices, self.hl, self.signal_features = self.process_data()
        
        return self.get_observation(), self.history


    def step(self, action):
        

        if self.current_tick >= self.end_tick:
            self.truncated = True
            self.done = True
        else:
            self.current_tick += 1
            
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        
        # check numpy if any nan values in self.debug_rewards
        
        # self.debug_rewards.append(step_reward)
        # if np.isnan(self.debug_rewards).any():
        #     print(self.debug_rewards)
        
        profit = self.calculate_profit(action)
        self.total_profit += profit

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
        
            
        self.position_history.append(self.position)
        info = dict(
            total_reward = self.total_reward,
            total_profit = self.total_profit,
            position = self.position.value
        )
        self.update_history(info)

        observation = self.get_observation()
        
        return observation, step_reward, self.done, self.truncated, info


    def get_observation(self):
        # return self.signal_features[(self.current_tick - self.window_size + 1):self.current_tick + 1]
        start_tick = self.current_tick - self.window_size + 1
        end_tick = self.current_tick + 1
        return np.append(np.append(self.signal_features[start_tick : end_tick].flatten(), self.position.value), self.current_pnl)


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
        def plot_positions(position_history, prices):
            short_ticks = []
            long_ticks = []
            for i, tick in enumerate(window_ticks):
                if position_history[i] == Positions.Short:
                    short_ticks.append(tick)
                elif position_history[i] == Positions.Long:
                    long_ticks.append(tick)

            plt.plot(short_ticks, prices[short_ticks], 'ro')
            plt.plot(long_ticks, prices[long_ticks], 'go')

        window_ticks = np.arange(len(self.position_history))
        
        plt.figure(figsize=(20, 8))
        plt.cla()
        
        plt.plot(self.prices)
        plot_positions(self.position_history, self.prices)
        
        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward + ' ~ ' +
            "Total Profit: %.6f" % self.total_profit + ' ~ '
        )

        plt.title('Continuous Test Environment')
        plt.show()
        
        
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
