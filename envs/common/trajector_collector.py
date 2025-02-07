"""
TODO:
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, DefaultDict
import numpy as np
import pickle
import numpy as np


@dataclass
class Trajectory:
    thread_id: int
    episode_id: int = field(default=int)
    episode_length: int = field(default=int)
    episode_reward: float = field(default=float)
    episode_radii: List[float] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    observations: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dense_states: DefaultDict[int, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    dense_time: List[np.ndarray] = field(default_factory=list)
        

class TrajectorCollector:
    episode_count : int = 0

    def __init__(self,
                 save_file_name: str = "try",
                 num_envs: int = 1,
                 state_index: List[int] = [0, 1]):
        
        self._save_file_name  = f"{save_file_name}_trajectories.pkl"
        self.trajectories = []
        self._num_envs = num_envs
        self._state_index = state_index
        for tid in range(num_envs):
            trajectory = Trajectory(thread_id=tid)
            self.trajectories.append(trajectory)

    def step(self, observations, actions, rewards, dense_index, dense_time, dense_states):
        for tid in range(self._num_envs):
            self.trajectories[tid].actions.append(actions[tid].copy())
            self.trajectories[tid].observations.append(observations[tid].copy())
            self.trajectories[tid].rewards.append(rewards[tid])
            self.trajectories[tid].dense_time.append(dense_time[:dense_index[tid], tid].copy())
            for i in self._state_index:
                self.trajectories[tid].dense_states[i].append(dense_states[:dense_index[tid], i, tid].copy())

    def end_episode(self, env_ids, final_observation, episode_length, episode_reward):
        for i, tid in enumerate(env_ids):
            self.trajectories[tid].episode_id = TrajectorCollector.episode_count
            self.trajectories[tid].episode_length = episode_length[i]
            self.trajectories[tid].episode_reward = episode_reward[i]
            self.trajectories[tid].observations.append(final_observation[i].copy())
            TrajectorCollector.episode_count +=1

        with open(self._save_file_name, 'ab') as f:
            for tid in env_ids:
                pickle.dump(self.trajectories[tid], f)

        for tid in env_ids:
            self.trajectories[tid] = Trajectory(thread_id=tid)

            


