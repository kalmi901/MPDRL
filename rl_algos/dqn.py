import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import time
import numpy as np
from typing import Any, Dict, List

try:
    from .common import ExperienceBuffer
except:
    from rl_algos.common import ExperienceBuffer


# Neural Networks -------------
# TODO:  


# 
    
class DQN():
    def __init__(self,
                 venv: Any,
                 learning_rate: float,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 exploration_noise: float = 0.1,
                 exploration_decay_rate: float = 0.0,
                 learning_starts: int = int(25e3),
                 gradient_steps: int = 4,
                 num_rollouts_per_env: int = 1,
                 seed: int = 42,
                 torch_deterministic: bool = True,
                 cuda: bool = True,
                 buffer_device: str = "cpu",
                 net_archs: Dict = None) -> None:
       
        # Seeding
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        # Attributes ----------------------------------
        self.venv = venv
        self.num_envs = venv.num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.exploration_decay_rate = exploration_decay_rate
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.num_rollouts_per_env = num_rollouts_per_env

        # Replay Memory --------------------------------
        self.memory = ExperienceBuffer(self.num_envs,
                                       self.buffer_size,
                                       self.venv.single_observation_space_shape,
                                       self.venv.single_action_space_shape,
                                       buffer_device,
                                       torch.float64)
        
        # Neural Networks ------------------------------
        if net_archs is None:
            self.q_network = None

        if net_archs is not None:
            pass

        # Syncronize the networks -----------------------


        # Optimizers--------------------


    def learn(self, total_number_of_steps: int, eval_frequency: int = None, tf_log: str = None):

        global_steps = 0
        num_updates = 0
        should_stop = False

        # ALGO ----------------------------
        obs, _ = self.venv.reset(seed=self.seed)

        try:
            while global_steps < total_number_of_steps:
                for _ in range(self.num_rollouts_per_env):
                    if global_steps < self.learning_starts:
                        actions = self.venv.action_space.sample()
                    else:
                        if torch.rand(1) < self.exploration_noise:
                            actions = self.venv.action_space.sample()
                        else:
                            with torch.no_grad():
                                q_values = self.q_network(obs)
                                actions = torch.argmax(q_values, dim=1)

                    if isinstance(actions, np.ndarray):
                        actions = torch.tensor(actions, device=self.device)

                    next_obs, rewards, terminated, time_outs, infos = self.venv.step(actions)

                    # Process Final Observation
                    real_next_obs = next_obs.clone()
                    if 'final_observation' in infos.keys():
                        real_next_obs[infos['dones']] = infos['final_observation']

                    self.memory.store_transitions(self.num_envs, obs, real_next_obs, actions, rewards, terminated)

                    global_steps += self.num_envs

                    obs = next_obs.clone()


                

        except KeyboardInterrupt:
            should_stop = True

        finally:
            self.venv.close()