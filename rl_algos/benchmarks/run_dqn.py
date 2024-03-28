from root import *
import gymnasium as gym
from gym_sync_env import ModifiedSyncVectorEnv


from rl_algos import DDPG


def make_env(env_id, seed, idx, capture_video, run_name):
    def wrapper():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, os.path.join(WORK_DIR, "videos", f"{run_name}"))
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return wrapper


if __name__ == "__main__":
    print("run")

    venvs = ModifiedSyncVectorEnv(
        [make_env("CartPole-v1", 1 + i, i, True, "test-runs") for i in range(4)]
    )


    model = DDPG(venvs)

    model.learn(total_number_of_steps=int(1e9))
