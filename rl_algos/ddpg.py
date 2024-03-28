import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from typing import List, Dict, Any


try:
    from .common import ExperienceBuffer
    from .common import TFWriter, WandbWriter
    from .common import build_torch_network
    from .common import process_final_observation
except: 
    from rl_algos.common import ExperienceBuffer
    from rl_algos.common import TFWriter, WandbWriter
    from rl_algos.common import build_torch_network
    from rl_algos.common import process_final_observation


# Neural Networks ----------
class Critic(nn.Module):
    def __init__(self, 
                 single_observation_space_shape: tuple,
                 single_action_space_shape: tuple,
                 fc_dims: List[int] = None,
                 fc_acts: List[str] = None) -> None:
        super().__init__()

        if (fc_dims is not None and
            fc_acts is not None and 
            len(fc_dims) == len(fc_acts)):
            print("Critic is created with custom parameters")
            self.qf = build_torch_network(input_dim=np.array(single_observation_space_shape).prod() \
                                                   +np.prod(single_action_space_shape),
                                          output_dim=1,
                                          fc_dims=fc_dims,
                                          fc_acts=fc_acts)

        else:
            print("Critic is created with default parameters")
            self.qf = nn.Sequential(
                nn.Linear(np.array(single_observation_space_shape).prod() + np.prod(single_action_space_shape), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, x:torch.Tensor, a:torch.Tensor):
        return self.qf(torch.cat([x, a], 1))


class Actor(nn.Module):
    def __init__(self,
                 single_observation_space_shape: tuple,
                 single_action_space_shape: tuple,
                 action_space_high: torch.Tensor,
                 action_space_low: torch.Tensor,
                 fc_dims: List[int] = None,
                 fc_acts: List[str] = None) -> None:
        super().__init__()

        if(fc_dims is not None and
           fc_acts is not None and
           len(fc_dims) == len(fc_acts)):
            print("Actor is created with custom parameters")
            self.pi = build_torch_network(input_dim=np.array(single_observation_space_shape).prod(),
                                          output_dim=np.prod(single_action_space_shape),
                                          fc_dims=fc_dims,
                                          fc_acts=fc_acts,
                                          last_act="Tanh")

        else:
            print("Actor is created with default parameters")
            self.pi =  nn.Sequential(
                nn.Linear(np.array(single_observation_space_shape).prod(), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, np.prod(single_action_space_shape)),
                nn.Tanh()
            )

        self.register_buffer(
            "action_scale", 
            torch.tensor((action_space_high - action_space_low) / 2.0 )
            )
        
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space_high + action_space_low) / 2.0)
            )
        
    def forward(self, x: torch.Tensor):
        return self.pi(x) * self.action_scale + self.action_bias 



class DDPG():
    metadata = {"hyperparameters": ["learning_rate", "gamma", "tau", "buffer_size", "batch_size", 
                                    "exploration_noise", "exploration_decay_rate", "learning_starts",
                                    "policy_frequency", "rollout_steps", "gradient_steps", "num_envs",
                                    "noise_clip", "seed", "torch_determnistic", "cuda", "storage_device"]}

    def __init__(self,
                 venvs: Any,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 exploration_noise: float = 0.1,
                 exploration_decay_rate: float = 0.0,
                 learning_starts: int = int(25e3),
                 policy_frequency: int = 2,
                 rollout_steps: int = 1,
                 gradient_steps: int = 1,
                 noise_clip: float = 0.5,
                 seed: int = 1,
                 torch_deterministic: bool = True,
                 cuda: bool = True,
                 storage_device: str= "cpu",
                 net_archs: Dict = None
                 ) -> None:
         
        # Seeding ------------------------------------
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        # Attributes ------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.venvs = venvs
        self.num_envs = venvs.num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.exploration_decay_rate = exploration_decay_rate
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.gradient_steps = gradient_steps
        self.rollout_steps = rollout_steps
        self.noise_clip = noise_clip            # Not implemented
        self.storage_device = storage_device

        # Neural networks ---------------------------------
        if net_archs is None:
            # Actor Network (Policy)
            self.pi = Actor(venvs.single_observation_space_shape, venvs.single_action_space_shape, 
                            venvs.single_action_space_high, venvs.single_action_space_low).to(self.device)
            self.pi_target = Actor(venvs.single_observation_space_shape, venvs.single_action_space_shape,
                                   venvs.single_action_space_high, venvs.single_action_space_low).to(self.device)

            # Critic Network (Action-Value Function)
            self.qf = Critic(venvs.single_observation_space_shape, venvs.single_action_space_shape).to(self.device)
            self.qf_target = Critic(venvs.single_observation_space_shape, venvs.single_action_space_shape).to(self.device)
        elif net_archs is not None:
            # Actor Network (Policy)
            self.pi = Actor(venvs.single_observation_space_shape, venvs.single_action_space_shape,
                            venvs.single_action_space_high, venvs.single_action_space_low,
                            net_archs["pi"][0], net_archs["pi"][1]).to(self.device)
            self.pi_target = Actor(venvs.single_observation_space_shape, venvs.single_action_space_shape,
                                   venvs.single_action_space_high, venvs.single_action_space_low,
                                   net_archs["pi"][0], net_archs["pi"][1]).to(self.device)

            # Critic Network (Action-Value Function)
            self.qf = Critic(venvs.single_observation_space_shape, venvs.single_action_space_shape,
                             net_archs["qf"][0], net_archs["qf"][1]).to(self.device)
            self.qf_target = Critic(venvs.single_observation_space_shape, venvs.single_action_space_shape,
                                    net_archs["qf"][0], net_archs["qf"][1]).to(self.device)

        # Syncronize the networks (qf_target = qf, pi_target = pi)
        self.pi_target.load_state_dict(self.pi.state_dict())
        self.qf_target.load_state_dict(self.qf.state_dict())

        # Set the optimizers (TODO: Implement Different learning rates!)
        #self.qf_optim = optim.Adam(self.qf.parameters(), lr=learning_rate)
        #self.pi_optim = optim.Adam(self.pi.parameters(), lr=learning_rate)
        
        self.qf_optim = optim.AdamW(self.qf.parameters(), lr=learning_rate)
        self.pi_optim = optim.AdamW(self.pi.parameters(), lr=learning_rate)

        print("-----Actor-----")
        print(self.pi)
        print("-----Critic----")
        print(self.qf)


        self.memory = ExperienceBuffer(
                                self.num_envs,
                                self.buffer_size,
                                self.venvs.single_observation_space_shape,
                                self.venvs.single_action_space_shape,
                                self.storage_device)      



    def predict(self, total_timesteps: int):
        try:
            obs, _ = self.venvs.reset(seed=self.seed)
            for global_step in range(0, total_timesteps, self.num_envs):
                with torch.no_grad():
                    actions = self.pi(obs)
                obs, _, _, _, infos = self.venvs.step(actions)
                
                if 'final_observation' in infos.keys():                        
                    for idx in range(len(infos['dones'])):
                        print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]}")
        
        except KeyboardInterrupt:
            pass
        finally:            
            self.venvs.close()


    def learn(self, total_timesteps: int, log_dir:str = None, project_name: str = None, trial_name: str = None, use_wandb: bool = False, log_frequency: int = 100):
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
            should_stop = False
            start_time = time.time()
            #qf_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.qf_optim, max_lr=self.learning_rate, total_steps=total_timesteps)
            #pi_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.pi_optim, max_lr=self.learning_rate, total_steps=total_timesteps)

            # ALGO ----------------
            obs, _ = self.venvs.reset(seed=self.seed)
            while global_step < total_timesteps:
                # ALGO LOGIC: rollout
                sampling_start = time.time()
                for _ in range(self.rollout_steps):
                    # ALGO LOGIC: action logic
                    if global_step < self.learning_starts:
                        actions = self.venvs.action_space.sample()
                        if not isinstance(actions, torch.Tensor):
                            actions = torch.tensor(actions, device=self.device)
                    else:
                        with torch.no_grad():
                            #actions = self.pi(torch.tensor(obs).to(self.device))
                            actions = self.pi(obs)
                            actions += torch.normal(0, self.pi.action_scale * self.exploration_noise)
                            actions.clamp(self.venvs.single_action_space_low.to(self.device), self.venvs.single_action_space_high.to(self.device))
                            #actions = actions.cpu().numpy().clip(self.venvs.single_action_space.low, self.venvs.single_action_space.high)
                            self.exploration_noise = max(self.exploration_noise -self.exploration_decay_rate, 0.0)
                    global_step += 1 * self.num_envs

                    # Execute the game and log data
                    next_obs, rewards, terminateds, _, infos = self.venvs.step(actions)

                    # Sava data to replay buffer and handle terminal observations
                    real_next_obs = process_final_observation(next_obs, infos)
                    if 'final_observation' in infos.keys():                        
                        for idx in range(len(infos['dones'])):
                            print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]}")
                            if log_data:
                                writer.add_scalar("charts/episode_return", infos["episode_return"][idx], global_step)
                                writer.add_scalar("charts/episode_length", infos["episode_length"][idx], global_step)

                    self.memory.store_transitions(self.num_envs, obs, real_next_obs, actions, rewards, terminateds)

                    # State transition
                    obs = next_obs.clone()

                # ALGO LOGIC: training -------------------------
                training_start = time.time()
                if global_step > self.learning_starts:
                    for _ in range(self.gradient_steps):
                        num_updates += 1
                        # Get sample from the memory
                        b_obs,      \
                        b_next_obs, \
                        b_actions,  \
                        b_rewards,  \
                        b_dones     = self.memory.sample(self.batch_size)

                        with torch.no_grad():
                            next_state_actions = self.pi_target(b_next_obs)
                            qf_next_target = self.qf_target(b_next_obs, next_state_actions).view(-1)
                            qf_target_values = b_rewards + (1.0 - b_dones) * self.gamma * qf_next_target
                        
                        qf_a_values = self.qf(b_obs, b_actions).view(-1)
                        qf_loss = F.mse_loss(qf_a_values, qf_target_values)

                        # Optimize the Q-network
                        self.qf_optim.zero_grad()
                        qf_loss.backward()
                        # TODO: Max grad norm?
                        self.qf_optim.step()

                        # Delayed policy updates
                        if num_updates % self.policy_frequency == 0:
                            pi_loss = -self.qf(b_obs, self.pi(b_obs)).mean()
                            
                            # Optimize the actor
                            self.pi_optim.zero_grad()
                            pi_loss.backward()
                            self.pi_optim.step()

                            # Soft update of the target network's weights
                            # θ′ ← τ θ + (1 −τ )θ′
                            for pi_param, pi_target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                                pi_target_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * pi_target_param.data)

                            for qf_param, qf_target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
                                qf_target_param.data.copy_(self.tau * qf_param.data + (1.0 - self.tau) * qf_target_param.data)

                            # Step schedulers
                            #qf_lr_scheduler.step()
                            #pi_lr_scheduler.step()

                        if log_data and (num_updates % log_frequency == 0):
                            writer.add_scalar("charts/learning_rate_pi", self.pi_optim.param_groups[0]["lr"], global_step)
                            writer.add_scalar("charts/learning_rate_qf", self.qf_optim.param_groups[0]["lr"], global_step)
                            writer.add_scalar("charts/exploration_noise", self.exploration_noise, global_step)
                            writer.add_scalar("losses/qf_loss", qf_loss.mean().item(), global_step)
                            writer.add_scalar("losses/pi_loss", pi_loss.item(), global_step)
                            writer.add_scalar("losses/qf1_values_mean", qf_a_values.mean().item(), global_step)
                            training_time = time.time() - sampling_start
                            sampling_time = training_start - sampling_start

                            writer.add_scalar("perf/training_time", training_time, global_step)
                            writer.add_scalar("perf/sampling_time", sampling_time, global_step)
                            writer.add_scalar("perf/DPS", float((global_step) / (time.time() - start_time)), global_step)

                # TODO: implement evaluation and early stopping here!!!
                


                if should_stop:
                    break
        except KeyboardInterrupt:
            should_stop = True
        
        finally:
            self.venvs.close()
            if log_data:
                writer.close()
                
