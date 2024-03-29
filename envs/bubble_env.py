"""
A minimalist interface to define vectorized bubble environments
"""


import torch
from abc import ABC, abstractmethod
from typing import Any, List, Dict

from .common import SpaceType
from .common import VSpace
from .common import Box
from .common import Discrete
from .common import Hybrid

class ActionSpaceDict:
    PA = {"IDX": List[int],
          "MIN": List[float],
          "MAX": List[float],
          "TYPE": str,
          "RES": int }
    
    PS = {"IDX": List[int],
          "MIN": List[float],
          "MAX": List[float],
          "TYPE": str,
          "RES": int }
    
    FR = {"IDX": List[int],
          "MIN": List[float],
          "MAX": List[float],
          "TYPE": str,
          "RES": int }


    def __init__(self, PA: Dict = None, PS: Dict = None, FR: Dict= None) -> None:
        self.PA = PA
        self.PS = PS
        self.FR = FR




class BubbleGPUEnv(ABC):

    solver: Any
    action_space_dict: ActionSpaceDict
    action_space: VSpace
    observation_space: VSpace

    def __init__(self, 
                 num_envs: int,
                 action_space_dict: ActionSpaceDict,
                 seed=int) -> None:
        
        
        self.num_envs   = num_envs
        self.seed       = seed
    

        self._create_observation_space()
        self._create_action_space(action_space_dict)
            


    def _create_observation_space(self):
        pass

    def _create_action_space(self, action_space_dict: ActionSpaceDict):
        low, high = [], []
        if action_space_dict is not None:
            if action_space_dict.PA["TYPE"] == "Box":
                pa_min = [action_space_dict.PA["MIN"][i] for i in range(len(action_space_dict.PA["IDX"]))]
                pa_max = [action_space_dict.PA["MAX"][i] for i in range(len(action_space_dict.PA["IDX"]))]
                low += pa_min
                high+= pa_max
            else:
                pass
        if action_space_dict.PS is not None:
            if action_space_dict.PS["TYPE"] == "Box":
                ps_min = [action_space_dict.PS["MIN"][i] for i in range(len(action_space_dict.PS["IDX"]))]
                ps_max = [action_space_dict.PS["MAX"][i] for i in range(len(action_space_dict.PS["IDX"]))]
                low += ps_min
                high+= ps_max
            else:
                pass

        self.action_space = Box(low=torch.Tensor(low),
                                high=torch.Tensor(high),
                                num_envs=self.num_envs,
                                dtype=torch.float32,
                                seed=self.seed)




    def step(self, action: torch.Tensor=None):
        if action is not None:
            self.actual_action = action
        self._set_action()
        self.solver.solve_my_ivp()
        self._get_observations()
        self._get_rewards()
        print("Step: Done")
    
    @abstractmethod
    def reset(self, seed: int, **options):
        raise NotImplementedError
    
    @abstractmethod
    def _set_action(self):
        raise NotImplementedError

    @abstractmethod
    def _get_rewards(self):
        raise NotImplementedError
    
    @abstractmethod
    def _get_observations(self):
        raise NotImplementedError
    
    @abstractmethod
    def _fill_control_parameters(self):
        raise NotImplementedError
    
    @abstractmethod
    def _fill_shared_parameters(self):
        raise NotImplementedError
    
    @abstractmethod
    def _fill_dynamic_parameters(self):
        raise NotImplementedError