from envs import Pos2B1D
import torch


if __name__ == "__main__":
    venvs = Pos2B1D(16, target_position="random", render_env=True)

    obs, _ = venvs.reset()
    #venvs.reset_envs([0, 4, 5])

    for i in range(5000):
        obs, _, _, _, _, venvs.step()
        #print(obs)
        print(torch.tensor(venvs.solver.get_device_array("actual_state", 2), device="cuda"))
