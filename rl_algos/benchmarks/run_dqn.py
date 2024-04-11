from root import *
import argparse
import os
import time
from distutils.util import strtobool

import gymnasium as gym
from gym_sync_env import ModifiedSyncVectorEnv


from rl_algos import DQN


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
    parser.add_argument("--use_wandb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="TryWANDB",
        help="the wandb's project name")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(1e4),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1,
        help="target smoothing coefficient (default: 1 - no smoothing)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-rate", type=float, default=0.05,
        help="the starting epsilon for epsilon-Greedy")
    parser.add_argument("--exploration-rate-min", type=float, default=0.05,
        help="the ending epsilon for epsilon-Greedy")
    parser.add_argument("--exploration-decay-rate", type=float, default=0.0,
        help=" the rate at which the exploration factor decreases over training")
    parser.add_argument("--learning-starts", type=int, default=1e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=100,
        help="the frequency of target policy update")
    parser.add_argument("--rollout-steps", type=int, default=10,
        help="the number of steps per environment between traning loops")
    parser.add_argument("--gradient-steps", type=int, default=1,
        help="the number of gradient descent steps")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--net-archs", type=str, default="128-128-ReLU",
        help="network architecture (64-64, 128-128, 256-256, 400-300) and activation function (ReLU, Tanh)")
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



NET_ARCHS = {"64-64-ReLU"  : [[  64,  64], ["ReLU", "ReLU"]],
             "128-128-ReLU": [[ 128, 128], ["ReLU", "ReLU"]],
             "256-256-ReLU": [[ 256, 256], ["ReLU", "ReLU"]],
             "400-300-ReLU": [[ 400, 300], ["ReLU", "ReLU"]],
             "64-64-Tanh"  : [[  64,  64], ["Tanh", "Tanh"]],
             "128-128-Tanh": [[ 128, 128], ["Tanh", "Tanh"]],
             "256-256-Tanh": [[ 256, 256], ["Tanh", "Tanh"]],
             "400-300-Tanh": [[ 400, 300], ["Tanh", "Tanh"]]}


if __name__ == "__main__":
    args = parse_args()
    project_name = args.wandb_project_name
    trial_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir =  WORK_DIR

    venvs = ModifiedSyncVectorEnv(
        [make_env(args.env_id, 1 + i, i, args.capture_video, trial_name) for i in range(args.num_envs)],
        action_space_type="Discrete",
        observation_space_type="Box"
    )

    net_arch = DQN.default_net_arch.copy()
    net_arch["hidden_dims"] = NET_ARCHS[args.net_archs][0]
    net_arch["activations"] = NET_ARCHS[args.net_archs][1]

    model = DQN(venvs,
                learning_rate = args.learning_rate,
                gamma = args.gamma,
                tau = args.tau,
                buffer_size = args.buffer_size,
                batch_size = args.batch_size,
                exploration_rate = args.exploration_rate,
                exploration_rate_min = args.exploration_rate_min,
                exploration_decay_rate = args.exploration_decay_rate,
                learning_starts = args.learning_starts,
                policy_frequency = args.policy_frequency,
                rollout_steps = args.rollout_steps,
                gradient_steps = args.gradient_steps,
                max_grad_norm = args.max_grad_norm,
                seed = args.seed,
                torch_deterministic = args.torch_deterministic,
                cuda = args.cuda,
                net_archs = net_arch)

    #model.learn(args.total_timesteps)
    model.learn(args.total_timesteps, log_dir, project_name, trial_name, args.use_wandb, 1)
