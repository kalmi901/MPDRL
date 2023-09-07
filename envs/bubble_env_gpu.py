import numpy as np
import numba as nb
import torch as th


import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
from typing import Union, List, Dict


class BubbleVecEnv(gym.vector.VectorEnv):
    def __init__(self,
                 num_envs: int,
                 single_action_space: Union[Box, Discrete, Tuple],
                 single_observation_space: Union[Box, Discrete, Tuple] ) -> None:
        
        super().__init__(num_envs, single_observation_space, single_action_space)



