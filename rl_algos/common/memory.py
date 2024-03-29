import torch
import numpy as np

class RolloutBuffer():
    def __init__(self,
                 num_envs: int,
                 rollout_steps: int,
                 single_observation_space_shape: tuple,
                 single_action_space_shape: tuple,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32) -> None:
        
        self.device = device
        self.dtype = dtype

        # Data Containers
        self.observations = torch.zeros((rollout_steps, num_envs) + single_observation_space_shape, dtype=self.dtype, device=self.device)
        self.actions      = torch.zeros((rollout_steps, num_envs) + single_action_space_shape, dtype=dtype, device=self.device)
        self.logprobs     = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=device)
        self.rewards      = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)
        self.values       = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)
        self.dones        = torch.zeros((rollout_steps, num_envs), dtype=self.dtype, device=self.device)


class ExperienceBuffer():
    def __init__(self,
                 num_envs: int,
                 buffer_size: int,
                 single_observation_space_shape: tuple,
                 single_action_space_shape: tuple,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32) -> None:
        
        
        assert buffer_size > num_envs, print("Buffer size must be higher than the number of environments!")
        
        self.buffer_size = buffer_size
        self.mem_counter = 0
        self.idx0        = 0
        self.device = device
        self.dtype = dtype

        # Data Containers
        self.observations     = torch.zeros((self.buffer_size, ) + single_observation_space_shape, dtype=self.dtype, device=self.device)
        self.new_observations = torch.zeros((self.buffer_size, ) + single_observation_space_shape, dtype=self.dtype, device=self.device)
        self.actions          = torch.zeros((self.buffer_size, ) + single_action_space_shape, dtype=self.dtype, device=self.device)
        self.rewards          = torch.zeros((self.buffer_size, ), dtype=self.dtype, device=self.device)
        self.dones            = torch.zeros((self.buffer_size, ), dtype=self.dtype, device=self.device)

    def store_transitions(self,
                        num_envs: int,
                        observations: torch.Tensor,
                        new_observations: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        dones: torch.Tensor):

        idxN = (self.idx0 + num_envs) % self.buffer_size
        if idxN > self.idx0:
            # Fill new data
            self.observations[self.idx0:idxN]        = observations.to(self.device)
            self.new_observations[self.idx0:idxN]    = new_observations.to(self.device)
            self.actions[self.idx0:idxN]             = actions.to(self.device)
            self.rewards[self.idx0:idxN]             = rewards.to(self.device)
            self.dones[self.idx0:idxN]               = dones.to(self.device)
        else:
            # Circular copy
            idxS = num_envs - idxN
            self.observations[self.idx0:]            = observations[0:idxS].to(self.device)
            self.observations[0:idxN]                = observations[idxS:].to(self.device)
            self.new_observations[self.idx0:]        = new_observations[0:idxS].to(self.device)
            self.new_observations[0:idxN]            = new_observations[idxS:].to(self.device)
            self.actions[self.idx0:]                 = actions[0:idxS].to(self.device)
            self.actions[0:idxN]                     = actions[idxS:].to(self.device)
            self.dones[self.idx0:]                   = dones[0:idxS].to(self.device)
            self.dones[0:idxN]                       = dones[idxS:].to(self.device)
            

        self.idx0 = idxN
        self.mem_counter += num_envs


    def sample(self, batch_size:int, device: str = "cuda"):
        max_sample_size = min(self.mem_counter, self.buffer_size)
        b_idx = torch.sort(torch.randperm(max_sample_size)[0:batch_size], 0)[0].to(self.device)

        b_obs       = self.observations[b_idx].to(device)
        b_next_obs  = self.new_observations[b_idx].to(device)
        b_actions   = self.actions[b_idx].to(device)
        b_rewards   = self.rewards[b_idx].to(device)
        b_dones     = self.dones[b_idx].to(device)

        return b_obs, b_next_obs, b_actions, b_rewards, b_dones

