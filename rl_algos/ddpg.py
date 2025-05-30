import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from pandas import DataFrame as df
from typing import Dict, Any, Union


try:
    from .common import ExperienceBuffer
    from .common import TFWriter, WandbWriter
    from .common import ActorDeterministic as Actor
    from .common import Critic
    from .common import process_final_observation
except: 
    from rl_algos.common import ExperienceBuffer
    from rl_algos.common import TFWriter, WandbWriter
    from rl_algos.common import ActorDeterministic as Actor
    from rl_algos.common import Critic
    from rl_algos.common import process_final_observation



class DDPG():
    metadata = {"hyperparameters": ["learning_rate", "gamma", "tau", "buffer_size", "batch_size", 
                                    "exploration_noise", "exploration_decay_rate", "learning_starts",
                                    "policy_frequency", "rollout_steps", "gradient_steps", "num_envs",
                                    "noise_clip", "seed", "torch_determnistic", "cuda", "storage_device"]}

    default_net_arch = {
        "pi": {
            "hidden_dims": [256, 256],
            "activations": ["ReLU", "ReLU"] },
        "qf": {
            "hidden_dims": [256, 256],
            "activations": ["ReLU", "ReLU"] }}
    
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
                 net_archs: Dict = default_net_arch
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
        action_high = venvs.single_action_space.high if isinstance(venvs.single_action_space.high, torch.Tensor) else torch.tensor(venvs.single_action_space.high)
        action_low  = venvs.single_action_space.low  if isinstance(venvs.single_action_space.low , torch.Tensor) else torch.tensor(venvs.single_action_space.low)

        self.pi = Actor(
            input_dim=np.array(venvs.single_observation_space.shape).prod(),
            output_dim=np.prod(venvs.single_action_space.shape),
            activations=net_archs["pi"]["activations"],
            hidden_dims=net_archs["pi"]["hidden_dims"],
            action_high=action_high,
            action_low=action_low).to(self.device)
        
        self.pi_target = Actor(
            input_dim=np.array(venvs.single_observation_space.shape).prod(),
            output_dim=np.prod(venvs.single_action_space.shape),
            activations=net_archs["pi"]["activations"],
            hidden_dims=net_archs["pi"]["hidden_dims"],
            action_high=action_high,
            action_low=action_low).to(self.device)

        self.qf = Critic(
            input_dim=np.array(venvs.single_observation_space.shape).prod() \
                    +np.prod(venvs.single_action_space.shape),
            output_dim=1,
            activations=net_archs["qf"]["activations"],
            hidden_dims=net_archs["qf"]["hidden_dims"]).to(self.device)
        
        self.qf_target = Critic(
            input_dim=np.array(venvs.single_observation_space.shape).prod() \
                    +np.prod(venvs.single_action_space.shape),
            output_dim=1,
            activations=net_archs["qf"]["activations"],
            hidden_dims=net_archs["qf"]["hidden_dims"]).to(self.device)

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
                                self.venvs.single_observation_space.shape,
                                self.venvs.single_action_space.shape,
                                self.storage_device)      


    def save_model(self, save_fname: str, save_dir: Union[str, None] = None):
        if save_dir is not None:
            save_fname = f"{save_dir}/{save_fname}"
        torch.save(self.pi.state_dict(), save_fname+"_actor.th")
        torch.save(self.qf.state_dict(), save_fname+"_critic.th")

    def load_model(self, load_fname: str, load_dir: Union[str, None] = None):
        # TODO: Handle exceptions
        if load_dir is not None:
            load_fname = f"{load_dir}/{load_fname}"
        loaded_actor_state_dict = torch.load(load_fname+"_actor.th")
        loaded_critic_state_dict = torch.load(load_fname+"_critic.th")
        self.pi.load_state_dict(loaded_actor_state_dict)
        self.pi_target.load_state_dict(loaded_actor_state_dict)
        self.qf.load_state_dict(loaded_critic_state_dict)
        self.qf_target.load_state_dict(loaded_critic_state_dict)

    def predict(self, total_timesteps: Union[int, None] = None, total_episodes: Union[int, None] = None, save_dir: Union[str, None] = None, stat_fname: Union[str, None] = None):
        assert total_timesteps is None or total_episodes is None, "Err, set 'total_timesteps' or 'total_episodes'"
        try:
            obs, _ = self.venvs.reset(seed=self.seed)
            global_step = 0
            episodes = 0
            episode_returns, episode_lengths = [], []
            while True:
                with torch.no_grad():
                    actions = self.pi(obs)
                obs, _, _, _, infos = self.venvs.step(actions)
                
                if 'final_observation' in infos.keys():                        
                    for idx in range(len(infos['dones'])):
                        episodes +=1
                        print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]}")
                        # Collect episode statistics
                        episode_returns.append(infos['episode_return'][idx].item())
                        episode_lengths.append(infos["episode_length"][idx].item())

                global_step += self.num_envs        
                # Check for simulation ends
                if total_episodes is not None:
                    if episodes >= total_episodes:
                        break
                else:
                    if global_step > total_timesteps:
                        break

            # ------- Save Statistics ---------
            if len(episode_returns) > 0 and len(episode_lengths) > 0:
                episode_returns = np.array(episode_returns, dtype=np.float32)
                episode_lengths = np.array(episode_lengths, dtype=np.float32)
                avg_return = np.mean(episode_returns)
                std_return = np.std(episode_returns)
                avg_length = np.mean(episode_lengths)
                std_length = np.std(episode_lengths)

                print("-------------------------------------------")
                print(f"Number of Episodes: {len(episode_returns):.0f}")
                print(f"Episode Returns: {avg_return:.3f}+/-{std_return:.3f}")
                print(f"Episode Lengths: {avg_length:.3f}+/-{std_length:.3f}")
                print("-------------------------------------------")

                # -------- Save Data to CSV --------------
                if stat_fname is not None:
                    if save_dir is not None:
                        stat_fname = f"{save_dir}/{stat_fname}.csv"
                    df({
                        "episode_returns": episode_returns,
                        "episode_lengths": episode_lengths
                    }).to_csv(stat_fname, index=False)

        except KeyboardInterrupt:
            pass
        finally:            
            self.venvs.close()


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
            episodes    = 0
            train_loop  = 0     # number of training loops
            should_stop = False
            start_time = time.time()
            #qf_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.qf_optim, max_lr=self.learning_rate, total_steps=total_timesteps)
            #pi_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.pi_optim, max_lr=self.learning_rate, total_steps=total_timesteps)

            # ALGO ----------------
            obs, _ = self.venvs.reset(seed=self.seed)
            while global_step < total_timesteps:
                train_loop += 1
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
                            actions += torch.normal(0, self.pi.action_scale * self.exploration_noise).clamp(-self.noise_clip, self.noise_clip)
                            actions = torch.clamp(actions, self.venvs.single_action_space.low.to(self.device), self.venvs.single_action_space.high.to(self.device))
                            #actions = actions.cpu().numpy().clip(self.venvs.single_action_space.low, self.venvs.single_action_space.high)
                            self.exploration_noise = max(self.exploration_noise -self.exploration_decay_rate, 0.0)
                    # Execute the game and log data
                    next_obs, rewards, terminateds, _, infos = self.venvs.step(actions)

                    # Save data to replay buffer and handle terminal observations
                    real_next_obs = process_final_observation(next_obs, infos)
                    if 'final_observation' in infos.keys():                        
                        for idx in range(len(infos['dones'])):
                            episodes +=1
                            print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]:0.3f}, episode_length={infos['episode_length'][idx]:0.0f}")
                            if log_data:
                                writer.add_scalar("charts/episode_return", infos["episode_return"][idx], episodes)
                                writer.add_scalar("charts/episode_length", infos["episode_length"][idx], episodes)

                    self.memory.store_transitions(self.num_envs, obs, real_next_obs, actions, rewards, terminateds)

                    # State transition
                    global_step += self.num_envs
                    obs = next_obs.clone()

                # ALGO LOGIC: training -------------------------
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

                    training_ends = time.time()
                    if log_data and (train_loop % log_frequency == 0) and (num_updates > self.policy_frequency):
                        writer.add_scalar("charts/learning_rate_pi", self.pi_optim.param_groups[0]["lr"], global_step)
                        writer.add_scalar("charts/learning_rate_qf", self.qf_optim.param_groups[0]["lr"], global_step)
                        writer.add_scalar("charts/exploration_noise", self.exploration_noise, global_step)
                        writer.add_scalar("losses/qf_loss", qf_loss.mean().item(), global_step)
                        writer.add_scalar("losses/pi_loss", pi_loss.item(), global_step)
                        writer.add_scalar("accuracy/qf_values_mean", qf_a_values.mean().item(), global_step)
                        training_time = training_ends - training_start
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
                
