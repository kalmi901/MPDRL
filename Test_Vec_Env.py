from envs import Pos1B1D
import numpy as np


if __name__ == "__main__":
    venvs = Pos1B1D(512)

    venvs.reset()

    for ic in range(500):

        obs, reward, _, _, info =venvs.step(venvs.action_space.sample())
        venvs.render()


       
