from root import *
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
from gym_sync_env import ModifiedSyncVectorEnv
import numpy as np

from rl_algos import DDPG


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--use_wandb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="TryWANDB",
        help="the wandb's project name")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pendulum-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--rollout-steps", type=int, default=16,
        help="the number of steps per environment between traning loops")
    parser.add_argument("--gradient-steps", type=int, default=16,
        help="the number of gradient descent steps")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args
    


def make_env(env_id, seed, idx, capture_video, run_name):
    def wrapper():
        if capture_video and idx==0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, os.path.join(WORK_DIR, "videos", f"{run_name}"))
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return wrapper

if __name__ == '__main__':
    args = parse_args()
    project_name = args.wandb_project_name
    trial_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir =  os.path.join(WORK_DIR)

    # env setup
    venvs = ModifiedSyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, trial_name) for i in range(args.num_envs)]
    )


    model = DDPG(venvs=venvs,
                 learning_rate = args.learning_rate,
                 gamma = args.gamma,
                 tau = args.tau,
                 buffer_size = args.buffer_size,
                 batch_size = args.batch_size,
                 exploration_noise = args.exploration_noise,
                 learning_starts = args.learning_starts,
                 policy_frequency = args.policy_frequency,
                 rollout_steps = args.rollout_steps,
                 gradient_steps = args.gradient_steps, 
                 noise_clip = args.noise_clip,
                 seed = args.seed,
                 torch_deterministic = args.torch_deterministic,
                 cuda = args.cuda)
    
    
    #model.learn(total_timesteps=args.total_timesteps)
    model.learn(total_timesteps=args.total_timesteps, log_dir=log_dir, project_name=project_name, trial_name=trial_name, use_wandb=args.use_wandb)