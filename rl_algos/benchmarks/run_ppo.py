from root import *
import argparse
import os
import time
from distutils.util import strtobool

import gymnasium as gym
from gym_sync_env import ModifiedSyncVectorEnv

from rl_algos import PPO

def parse_args():
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
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--mini-batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rollout-steps", type=int, default=128,
        help="the number of steps per environment between traning loops")
    parser.add_argument("--num_update_epochs", type=int, default=12,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.00,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def wrapper():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, os.path.join(WORK_DIR, "videos", f"{run_name}"))
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return wrapper


if __name__ == "__main__":
    args = parse_args()
    project_name = args.wandb_project_name
    trial_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir =  os.path.join(WORK_DIR)


    # env setup
    venvs = ModifiedSyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, trial_name) for i in range(args.num_envs)]
    )


    model = PPO(venvs=venvs,
                learning_rate = args.learning_rate,
                gamma = args.gamma,
                gae_lambda = args.gae_lambda,
                mini_batch_size = args.mini_batch_size,
                clip_coef = args.clip_coef,
                clip_vloss = args.clip_vloss,
                ent_coef = args.ent_coef,
                vf_coef = args.vf_coef,
                max_grad_norm = args.max_grad_norm,
                target_kl = args.target_kl,
                norm_adv = args.norm_adv,
                rollout_steps = args.rollout_steps,
                num_update_epochs = args.num_update_epochs,
                seed = args.seed,
                torch_deterministic = args.torch_deterministic,
                cuda = args.cuda) 
    

    #model.learn(total_timesteps=args.total_timesteps)
    model.learn(total_timesteps=args.total_timesteps, log_dir=log_dir, project_name=project_name, trial_name=trial_name, use_wandb=args.use_wandb)