"""
A minimalist interface to define vectorized bubble environments
"""


import torch
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union

from .common import SpaceType
from .common import VSpace
from .common import Box
from .common import Discrete
from .common import Hybrid

class ActionSpaceDict:
    action_components = ["PA", "PS", "FR"]
    PA = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str,
          "RES" : int }
    
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


    def __init__(self, PA: Dict = None, PS: Dict = None, FR: Dict = None) -> None:
        self.PA = PA
        self.PS = PS
        self.FR = FR

class ObservationSpaceDict:
    observed_quatities = ["R0", "XT", "X", "V", "R", "U"]

    # Equilibrium bubble radious
    R0 = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str}
    
    # Target Position
    XT = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str}
    
    # Bubble Position
    X  = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str,
          "STACK": List[float]}
    
    # Bubble Translational Velocity
    V  = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str,
          "STACK": List[float]}
    
    # Bubble Radius
    R  = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str,
          "RES" : List[float],
          "STACK": List[float]}
    
    # Bubble Wall velocity
    U  = {"IDX" : List[int],
          "MIN" : List[float],
          "MAX" : List[float],
          "TYPE": str,
          "STACK": List[float]}
    

    def __init__(self, R0: Dict = None, XT: Dict = None, X: Dict = None,
                 V: Dict = None, R: Dict = None, U: Dict = None) -> None:
        self.R0 = R0
        self.XT = XT
        self.X  = X
        self.V  = V
        self.R  = R
        self.U  = U

class BubbleGPUEnv(ABC):

    solver: Any
    action_space_dict: ActionSpaceDict
    action_space: VSpace
    observation_space: VSpace

    def __init__(self, 
                 num_envs: int,
                 action_space_dict: ActionSpaceDict,
                 observation_space_dict = ObservationSpaceDict,
                 seed=int) -> None:
        
        
        self.num_envs   = num_envs
        self.seed       = seed
    

        self._create_action_space(action_space_dict)
        self._create_observation_space(observation_space_dict)
            

    def _create_observation_space(self, observation_space_dict):
        low, high = [], []
        attributes = ["R0", "XT", "X", "V", "R", "U"]
        for attr in attributes:
            observed_quantity = getattr(observation_space_dict, attr)
            if observed_quantity is not None: 
                if observed_quantity["TYPE"] == "Box":
                    if "STACK" in observed_quantity.keys():
                        low_tmp = [observed_quantity["MIN"][i] for _ in range(observed_quantity["STACK"]) \
                                   for i in range(len(observed_quantity["IDX"]))]
                        high_tmp= [observed_quantity["MAX"][i] for _ in range(observed_quantity["STACK"]) \
                                   for i in range(len(observed_quantity["IDX"]))]
                    else:
                        low_tmp = [observed_quantity["MIN"][i] for i in range(len(observed_quantity["IDX"]))]
                        high_tmp= [observed_quantity["MAX"][i] for i in range(len(observed_quantity["IDX"]))]
                            
                    low += low_tmp
                    high+= high_tmp                         
                else:
                    # TODO: Collect Discrete varibales here
                    pass

        self.observation_space =Box(low=torch.Tensor(low),
                                    high=torch.Tensor(high),
                                    num_envs=self.num_envs,
                                    dtype=torch.float32,
                                    seed=self.seed)


    def _create_action_space(self, action_space_dict: ActionSpaceDict):
        low, high = [], []
        if action_space_dict.PA is not None:
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
                # TODO: Add Frequency
                pass

        self.action_space = Box(low=torch.Tensor(low),
                                high=torch.Tensor(high),
                                num_envs=self.num_envs,
                                dtype=torch.float32,
                                seed=self.seed)



    @abstractmethod
    def step(self, action: torch.Tensor=None):
        raise NotImplementedError
        
    
    @abstractmethod
    def reset(self, seed: int, **options):
        raise NotImplementedError
    

    def close(self):
        print("Environment is closed")

