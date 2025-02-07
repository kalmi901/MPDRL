from envs import Pos2B1D, ObservationSpaceDict
import torch


if __name__ == "__main__":

    OBSERVATION_SPACE = ObservationSpaceDict(
    XT = {"IDX": [0, 1], "MIN": [-0.25, -0.25], "MAX": [0.25, 0.25], "TYPE": "Box"},
    X  = {"IDX": [0, 1], "MIN": [-0.25, -0.25], "MAX": [0.25, 0.25], "TYPE": "Box", "STACK": 2}
    )

    venvs = Pos2B1D(16,
                    R0=[60.0, 60.0],
                    freqs=[25.0, 50.0],
                    ac_type="SW_N", 
                    target_position=0.0, 
                    initial_position=0.0, 
                    initial_distace=0.2, 
                    final_distance=0.5,
                    rel_freq=25.0,
                    observation_space_dict=OBSERVATION_SPACE,
                    render_env=True,
                    collect_trajectories=True)

    obs, _ = venvs.reset()
    #venvs.reset_envs([0, 4, 5])

    """for i in range(16):
        print(f"CP[{i}]")
        print(torch.tensor(venvs.solver.get_device_array("control_parameters", i), device="cuda"))
        print(torch.tensor(venvs.solver.get_device_array("control_parameters", i+16), device="cuda"))
        print("------")
    
    for i in range(4):
        print(f"DP[{i}]")
        print(torch.tensor(venvs.solver.get_device_array("dynamic_parameters", i), device="cuda"))
    """
    action = torch.tile(torch.tensor([0.5, 0.0, 0.0, 0.0]), (venvs.num_envs, 1)).to("cuda")
    for i in range(500):
        obs, _, _, _, _, venvs.step(action)
        """print(obs)
        for i in range(4):
            print(f"DP[{i}]")
            print(torch.tensor(venvs.solver.get_device_array("dynamic_parameters", i), device="cuda"))
        input()
        """
