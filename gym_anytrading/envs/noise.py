import numpy as np

class NoiseGenerator:
    
    ########################################################
    # Noise from https://arxiv.org/pdf/2305.02882.pdf
    # Noise ranking: https://docs.google.com/spreadsheets/d/1CTZiRX_s9RQ3sHVq6WEmWh0SonE5xDEEyDRgmVrLWEs/edit?pli=1#gid=220110982 
    ########################################################
    
    @classmethod
    def random_uniform_scale_reward(cls, step_reward):
        noise_rate = 0.01
        low = 0.9
        high = 1.1
        if np.random.rand() <= noise_rate:
            step_reward *= np.random.uniform(low, high)
        return step_reward
    
    # custom noise, not in paper
    @classmethod
    def random_normal_scale_reward(cls, step_reward):
        noise_rate = 1
        mean = 1
        std = 1
        if np.random.rand() <= noise_rate:
            step_reward *= np.random.normal(mean, std)
        return step_reward
    
    @classmethod
    def random_normal_noise_reward(cls, step_reward):
        noise_rate = 1
        mean = 0
        std = 1.0
        if np.random.rand() <= noise_rate:
            step_reward += np.random.normal(mean, std)
        return step_reward
    
    @classmethod
    def random_uniform_noise_reward(cls, step_reward):
        noise_rate = 1
        low = -0.001
        high = 0.001
        if np.random.rand() <= noise_rate:
            step_reward += np.random.uniform(low, high)
        return step_reward


