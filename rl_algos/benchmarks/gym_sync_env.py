import gym as gym
from gymnasium import Env 
from gymnasium.vector import SyncVectorEnv
from typing import Callable, Iterable, List, Tuple


import torch
import numpy as np

class ModifiedSyncVectorEnv(SyncVectorEnv):
    def __init__(self, 
                 env_fns: Iterable[Callable[[], Env]], 
                 observation_space: gym.Space = None, 
                 action_space: gym.Space = None,
                 action_space_type: str = "Box",
                 observation_space_type: str = "Box",
                 copy: bool = True):
        
        super().__init__(env_fns, observation_space, action_space, copy)


        if action_space_type == "Box":
            self.single_action_space_shape       = self.single_action_space.shape
            self.single_action_space_low         = torch.tensor(self.single_action_space.low)
            self.single_action_space_high        = torch.tensor(self.single_action_space.high)

        if action_space_type == "Discre":
            self.single_action_space_shape       = (1, )
            self.single_action_space_shape       = torch.tensor(self.single_action_space)

        if observation_space_type == "Box":
            self.single_observation_space_shape  = self.single_observation_space.shape


    def reset(self, seed: int = None):
        return torch.tensor(super().reset(seed=seed)[0], device="cuda"), {}


    def step(self, actions):
        
        # Copy Tensors to the device!
        if isinstance(actions, torch.Tensor):
            next_obs, rewards, terminateds, truncateds, infos = super().step(actions.cpu().numpy())
        
        if isinstance(actions, np.ndarray):
            next_obs, rewards, terminateds, truncateds, infos = super().step(actions)

        next_obs    = torch.tensor(next_obs, device="cuda", dtype=torch.float32)
        rewards     = torch.tensor(rewards, device="cuda", dtype=torch.float32)
        terminateds = torch.tensor(terminateds, device="cuda", dtype=torch.float32)
        truncateds  = torch.tensor(truncateds, device="cuda", dtype=torch.float32)

        my_infos = {}
        if 'final_observation' in infos.keys():
            #infos['final_observations'] = torch.tensor(infos['final_observation'])
            dones = torch.logical_or(terminateds, truncateds)   # Trajectory segments end
            idx = torch.where(dones)[0]
            #print(idx)
            #real_next_obs[idx] = np.stack(infos['final_observation'][idx]).copy()   # last state of trajectory
            my_infos['final_observation'] = torch.tensor(np.stack(infos['final_observation'][idx.cpu().numpy()]), device="cuda")
            my_infos['dones'] = idx

            my_infos['episode_return'] = torch.tensor(np.hstack([info['episode']['r'] for info in infos['final_info'][idx.cpu().numpy()]]), device="cuda")
            my_infos['episode_length'] = torch.tensor(np.hstack([info['episode']['l'] for info in infos['final_info'][idx.cpu().numpy()]]), device="cuda")
            
               
        
        return next_obs, rewards, terminateds, truncateds, my_infos