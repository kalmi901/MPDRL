"""
A minimalist interface to define vectorized bubble environments
"""


from abc import ABC, abstractmethod
import torch
from typing import Any

from .common import VSpace


class BubbleGPUEnv(ABC):

    def __init__(self, 
                 num_envs: int,
                 single_action_space: VSpace = None,
                 single_observation_space: VSpace = None) -> None:
        
        
        self.num_envs = num_envs
        self.solver: Any
        self.action_space: VSpace
        self.observation_space: VSpace

        self._observation: torch.Tensor
        self._rewards: torch.Tensor
            

    @abstractmethod
    def step(self, action: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, seed: int, **options):
        raise NotImplemented
    
    @abstractmethod
    def _get_rewards(self):
        raise NotImplemented
    
    @abstractmethod
    def _get_observations(self):
        raise NotImplemented