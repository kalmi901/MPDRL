import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import time
import numpy as np
from typing import Any, Dict

try:
    from .common import ExperienceBuffer
    from .common import TFWriter, WandbWriter
    from .common import Critic, DualCritic
    from .common import process_final_observation
except: 
    from rl_algos.common import ExperienceBuffer
    from rl_algos.common import TFWriter, WandbWriter
    from rl_algos.common import Critic, DualCritic
    from rl_algos.common import process_final_observation


    
class DQN():
    metadata = {"hyperparameters" : ["learning_rate", "gamma", "tau", "buffer_size", "batch_size", "exploration_rate", "exploration_rate_min", "exploration_decay_rate",
                                     "learning_starts", "policy_frequency", "rollout_steps", "gradient_steps", "max_grad_norm", 
                                     "seed", "torch_deterministic", "cuda", "buffer_device"]}
    
    default_net_arch = {
            "hidden_dims" : [120, 84],
            "activations" : ["ReLU", "ReLU"],
            "shared_dims" : 0,
            "dual"        : True }

    def __init__(self,
                 venvs: Any,
                 learning_rate: float = 2.5e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 exploration_rate: float = 0.1,
                 exploration_rate_min: float = 0.1,
                 exploration_decay_rate: float = 0.0,
                 learning_starts: int = int(25e3),
                 policy_frequency: int = 2,
                 rollout_steps: int = 1,
                 gradient_steps: int = 4,
                 max_grad_norm: float = 0.5,
                 seed: int = 42,
                 torch_deterministic: bool = True,
                 cuda: bool = True,
                 buffer_device: str = "cpu",
                 net_archs: Dict = default_net_arch
                 ) -> None:
       
        # Seeding
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        # Attributes ----------------------------------
        self.venvs = venvs
        self.num_envs = venvs.num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_decay_rate = exploration_decay_rate
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.gradient_steps = gradient_steps
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.buffer_device = buffer_device

        # Replay Memory --------------------------------
        self.memory = ExperienceBuffer(self.num_envs,
                                       self.buffer_size,
                                       self.venvs.single_observation_space.shape,
                                       self.venvs.single_action_space.shape,
                                       self.buffer_device,
                                       torch.float32)
        
        # Neural Networks ------------------------------
        if net_archs["dual"]:
            self.qf = DualCritic(input_dim=np.array(venvs.single_observation_space.shape).prod(),
                             output_dim=venvs.single_action_space.n,
                             **net_archs).to(self.device)
            
            self.qf_target = DualCritic(input_dim=np.array(venvs.single_observation_space.shape).prod(),
                             output_dim=venvs.single_action_space.n,
                             **net_archs).to(self.device)
        else:
            self.qf = Critic(input_dim=np.array(venvs.single_observation_space.shape).prod(),
                             output_dim=venvs.single_action_space.n,
                             **net_archs).to(self.device)
            
            self.qf_target = Critic(input_dim=np.array(venvs.single_observation_space.shape).prod(),
                             output_dim=venvs.single_action_space.n,
                             **net_archs).to(self.device)  

        # Syncronize the networks -----------------------
        self.qf_target.load_state_dict(self.qf.state_dict())

        # Optimizers--------------------
        self.qf_optim = optim.AdamW(self.qf.parameters(), lr=learning_rate)

        print("-----Q-Network-----")
        print(self.qf)


    def learn(self, total_timesteps: int, log_dir:str = None, project_name: str = None, trial_name: str = None, use_wandb: bool = False, log_frequency: int = 10):
        try:
            log_data = False
            if all(var is not None for var in [log_dir, project_name, trial_name]):  
                if use_wandb:
                    writer = WandbWriter(log_dir=log_dir, project_name=project_name, run_name=trial_name, model=self)
                else:
                    writer = TFWriter(log_dir=log_dir, project_name=project_name, run_name=trial_name, model=self)
                log_data = True

            else:
                print("Warning: Log is not created.")

            global_step = 0     # timesteps
            num_updates = 0     # nn update
            train_loop  = 0     # number of training loops
            should_stop = False
            start_time = time.time()


            # ALGO ----------------------------
            obs, _ = self.venvs.reset(seed=self.seed)
            while global_step < total_timesteps:
                train_loop += 1
                # ALGO LOGIC: rollout
                sampling_start = time.time()
                for _ in range(self.rollout_steps):
                    # ALGO LOGIC: action logic: e-Greedy
                    if (global_step < self.learning_starts or 
                        torch.rand(1) < self.exploration_rate):
                        # ALGO: random sampling
                        actions = self.venvs.action_space.sample()
                        if not isinstance(actions, torch.Tensor):
                            actions = torch.tensor(actions, device=self.device)
                    else:
                        with torch.no_grad():
                            q_values = self.qf(obs)
                            actions = torch.argmax(q_values, dim=1)

                    # Execute the envrionment 
                    next_obs, rewards, terminated, _, infos = self.venvs.step(actions)

                    # Save data to replay buffer and handle terminal observation
                    real_next_obs = process_final_observation(next_obs, infos)
                    if 'final_observation' in infos.keys():
                        for idx in range(len(infos['dones'])):
                            print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]}")
                            if log_data:
                                writer.add_scalar("charts/episode_return", infos["episode_return"][idx], global_step)
                                writer.add_scalar("charts/episode_length", infos["episode_length"][idx], global_step)

                    self.memory.store_transitions(self.num_envs, obs, real_next_obs, actions, rewards, terminated)

                    # State transition
                    global_step += self.num_envs
                    obs = next_obs.clone()

                # ALGO LOGIC: training -------------
                if global_step > self.learning_starts:
                    training_start = time.time()
                    for _ in range(self.gradient_steps):
                        num_updates += 1
                        # Get sample from the memory
                        b_obs,      \
                        b_next_obs, \
                        b_actions,  \
                        b_rewards,  \
                        b_dones     = self.memory.sample(self.batch_size)

                        with torch.no_grad():
                            # DQN ----
                            # qf_next_target, _ = self.qf_target(b_next_obs).max(dim=1)
                            # Double DQN ------
                            a_next = self.qf(b_next_obs).argmax(1, keepdim=True)
                            qf_next_target = self.qf_target(b_next_obs).gather(1, a_next.long()).squeeze()
                            qf_target_values = b_rewards.flatten() + self.gamma * qf_next_target * (1.0 - b_dones.flatten())


                        qf_a_values = self.qf(b_obs).gather(1, b_actions.unsqueeze(1).long()).squeeze()
                        qf_loss = F.mse_loss(qf_a_values, qf_target_values)

                        # Optimize the Q-network
                        self.qf_optim.zero_grad()
                        qf_loss.backward()
                        nn.utils.clip_grad_norm_(self.qf.parameters(), self.max_grad_norm)
                        self.qf_optim.step()

                        # Delayed policy updates
                        if num_updates % self.policy_frequency == 0:
                            # Soft update of the target network's weights
                            # θ′ ← τ θ + (1 −τ )θ′

                            for qf_param, qf_target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
                                qf_target_param.data.copy_(self.tau * qf_param.data + (1.0 - self.tau) * qf_target_param.data)

                    training_ends = time.time()
                    if log_data and (train_loop % log_frequency == 0):
                        writer.add_scalar("charts/learning_rate_qf", self.qf_optim.param_groups[0]["lr"], global_step)
                        writer.add_scalar("charts/exploration_rate", self.exploration_rate, global_step)
                        writer.add_scalar("losses/qf_loss", qf_loss.mean().item(), global_step)
                        writer.add_scalar("accuracy/qf_values_mean", qf_a_values.mean().item(), global_step)
                        training_time = training_ends - training_start
                        sampling_time = training_start - sampling_start

                        writer.add_scalar("perf/training_time", training_time, global_step)
                        writer.add_scalar("perf/sampling_time", sampling_time, global_step)
                        writer.add_scalar("perf/DPS", float((global_step) / (time.time() - start_time)), global_step)

                    # TODO: implement evaluation and early stopping

                    if should_stop:
                        break

        except KeyboardInterrupt:
            should_stop = True

        finally:
            self.venvs.close()
            if log_data:
                writer.close()