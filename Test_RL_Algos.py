from rl_algos import DDPG
from envs import Pos1B1D

if __name__ == "__main__":
    venv = Pos1B1D(1024)
    model = DDPG(venv=venv)