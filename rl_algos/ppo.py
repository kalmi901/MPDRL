import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from typing import List, Dict, Any

try:
    from .common import RolloutBuffer
    from .common import TFWriter, WandbWriter
    from .common import build_torch_network
    from .common import process_final_observation
except:
    from rl_algos.common import RolloutBuffer
    from rl_algos.common import TFWriter, WandbWriter
    from rl_algos.common import build_torch_network
    from rl_algos.common import process_final_observation


# Neural Networks ---------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticPolicy(nn.Module):
    def __init__(self,
                 venvs: Any,
                 net_archs: Dict = None,
                 log_std_max: float = 4.0,
                 log_std_min: float = -20.0,
                 log_std_init: float = 1.0) -> None:
        super().__init__()

        self._log_std_max = log_std_max
        self._log_std_min = log_std_min

        self._log_std = nn.Parameter(torch.ones(np.prod(venvs.single_action_space.shape)) * log_std_init)


        if net_archs == None:
            # Create NN-s with default

            self.pi =  nn.Sequential(
                layer_init(nn.Linear(np.array(venvs.single_observation_space.shape).prod(), 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, np.prod(venvs.single_action_space.shape)), std=1),
            )

            self.vf =  nn.Sequential(
                layer_init(nn.Linear(np.array(venvs.single_observation_space.shape).prod(), 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 1)),
            )

        else:
            pass


    def forward(self, x: torch.Tensor, a: torch.Tensor = None):
        mean = self.pi(x)
        log_std = torch.ones_like(mean) * self._log_std.clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()

        probs = torch.distributions.Normal(mean, std)
        if a is None:
            a = probs.sample()

        return a, probs.log_prob(a).sum(1), probs.entropy().sum(1), self.vf(x)
        
  

class PPO():
    metadata = {"hyperparameters" : ["learning_rate", "gamma"]}

    def __init__(self,
                 venvs: Any,
                 learning_rate: float = 2.5e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 mini_batch_size: int = 256,
                 clip_coef: float = 0.2,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 norm_adv: bool = True,
                 rollout_steps: int = 32,
                 num_update_epochs: int = 4,
                 seed: int = 1,
                 torch_deterministic: bool = True,
                 cuda: bool = True,
                 storage_device: str = "cuda",
                 net_archs: Dict = None
                 ) -> None:
        
        # Seeding ------------------------
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        # Attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.venvs = venvs
        self.num_envs = venvs.num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.rollout_steps = rollout_steps
        self.mini_batch_size = mini_batch_size
        self.batch_size = self.num_envs * self.rollout_steps       # = buffer_size
        self.num_update_epochs = num_update_epochs
        self.iterations_per_epoch = self.batch_size // self.mini_batch_size + (0 if self.batch_size % self.mini_batch_size == 0 else 1) 
        self.gradient_steps = self.num_update_epochs * self.iterations_per_epoch
        self.storage_device = storage_device

        # Neural Networks ----------------------
        self.policy = ActorCriticPolicy(venvs).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

        # Rollout buffer -----------------------
        self.memory = RolloutBuffer(self.num_envs,
                                    self.rollout_steps,
                                    self.venvs.single_observation_space.shape,
                                    self.venvs.single_action_space.shape,
                                    self.storage_device)


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

            # ALGO --------------------
            obs, _ = self.venvs.reset(seed=self.seed)
            dones = torch.zeros(self.num_envs, device=self.device)
            while global_step < total_timesteps:
                # ALGO LOGIC: rollout
                sampling_start = time.time()
                for step in range(self.rollout_steps):
                    with torch.no_grad():
                        actions, logprobs, _, values = self.policy(obs)
                        actions.clamp(self.venvs.single_action_space_low.to(self.device), self.venvs.single_action_space_high.to(self.device))

                    # Execute the game and log data
                    next_obs, rewards, terminateds, time_outs, infos = self.venvs.step(actions)

                    # Save data to the buffer and handle terminal observations
                    real_next_obs = process_final_observation(next_obs, infos)
                    if 'final_observation' in infos.keys():                        
                        for idx in range(len(infos['dones'])):
                            print(f"global_step={global_step}, episode_return={infos['episode_return'][idx]}")
                            if log_data:
                                writer.add_scalar("charts/episode_return", infos["episode_return"][idx], global_step)
                                writer.add_scalar("charts/episode_length", infos["episode_length"][idx], global_step)

                    # Handle time-out --> add future value for the reward
                    idx = torch.where(time_outs)[0]
                    with torch.no_grad():
                        real_next_values = self.policy.vf(real_next_obs).view(-1)
                        if len(idx) > 0:
                            rewards[idx] += (self.gamma * real_next_values[idx])

                    # Store transition to memory
                    self.memory.store_transition(step, obs, actions, logprobs, rewards, values.view(-1), dones)

                    # State transition
                    dones = torch.logical_or(terminateds, time_outs)        # Trial trajectory ends
                    global_step += self.num_envs
                    obs = next_obs.clone()

                # ALGO LOGIC: training ----------------
                training_start = time.time()
                # GAE -----
                self.memory.compute_gae_estimate(real_next_values, dones, self.gamma, self.gae_lambda)
                # Get the sample from the replay memory
                b_obs,          \
                b_logprobs,     \
                b_actions,      \
                b_advantages,   \
                b_returns,      \
                b_values        = self.memory.sample()

                # shullfe data
                b_idx = torch.randperm(self.batch_size)

                clipfracs = []
                for _ in range(self.num_update_epochs):
                    for idx0 in range(0, self.batch_size, self.mini_batch_size):
                        idxN = min(idx0 + self.mini_batch_size, self.batch_size)
                        mb_idx = b_idx[idx0:idxN]
                        num_updates += 1

                        # TODO: Handle discrete action space!
                        #if isinstance(self.venvs.single_action_space, Discrete):
                        #    _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(b_obs[mb_idx], b_actions.long()[mb_idx])

                        _, newlogprob, entropy, newvalues = self.policy(b_obs[mb_idx], b_actions[mb_idx])
                        logratio = newlogprob - b_logprobs[mb_idx]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # approx_kl http://joschu.net/blog/kl-approx.html
                            #approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                        # Normalize the advanteges
                        mb_advantages = b_advantages[mb_idx]
                        if self.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Calculate the losses -----------
                        # Policy loss
                        pi_loss = torch.max(-mb_advantages * ratio,
                                           -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)).mean()
                        
                        # Value loss
                        newvalues = newvalues.view(-1)
                        if self.clip_vloss:
                            vf_loss_unclipped = (newvalues - b_returns[mb_idx])**2
                            vf_clipped = b_values[mb_idx] \
                                        + torch.clamp(newvalues - b_values[mb_idx],
                                                        -self.clip_coef,
                                                        +self.clip_coef)
                            vf_loss_clipped = (vf_clipped - b_returns[mb_idx])**2
                            vf_loss_max = torch.max(vf_loss_unclipped, vf_loss_clipped)
                            vf_loss = 0.5 * vf_loss_max.mean()
                        else:
                            vf_loss = 0.5 * ((newvalues - b_returns[mb_idx])**2).mean()

                        # Entropy loss
                        ent_loss = entropy.mean()

                        # Total loss function
                        loss = pi_loss - self.ent_coef * ent_loss + self.vf_coef * vf_loss

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                        if self.target_kl is not None and approx_kl > self.target_kl:
                            should_stop = True


                training_ends = time.time()
                # Calculate the explaine variance
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


                if log_data and (train_loop % log_frequency == 0):
                    writer.add_scalar("charts/learning_rate_pi", self.optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/learning_rate_vf", self.optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("losses/vf_loss", vf_loss.item(), global_step)
                    writer.add_scalar("losses/pi_loss", pi_loss.item(), global_step)
                    writer.add_scalar("losses/ent_loss", ent_loss.item(), global_step)
                    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                    writer.add_scalar("accuracy/vf_values_mean", newvalues.mean().item(), global_step)
                    writer.add_scalar("accuracy/approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar("accuracy/explained_variance", explained_var, global_step)

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