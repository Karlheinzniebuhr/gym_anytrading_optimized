from gymnasium.envs.registration import register
from copy import deepcopy

from . import datasets

register(
    id='CryptoEnv',
    entry_point='gym_anytrading.envs:CryptoEnv',
)

register(
    id='CryptoEnvContinuous',
    entry_point='gym_anytrading.envs:CryptoEnvContinuous',
)
