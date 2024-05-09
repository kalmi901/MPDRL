from envs import Pos1B1D
import numpy as np


if __name__ == "__main__":
    venvs = Pos1B1D(16, target_position="random")

    venvs.reset()

    venvs.reset_envs([0, 4, 5])

    for i in range(50):
        venvs.step()
        venvs.render()
        input()

    """
    for ic in range(500):

        obs, reward, _, _, info =venvs.step(venvs.action_space.sample())
        venvs.render()
    """

       
