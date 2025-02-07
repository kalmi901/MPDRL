import torch
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from typing import List, Optional, Union

from envs import BubbleGPUEnv
from envs import ActionSpaceDict
from envs import ObservationSpaceDict
from .common import TrajectorCollector

# SOLVER SPECIFIC PARAMETERS
# System definition
from gpu_numba_solver.system_definitions import KM1D
from gpu_numba_solver.system_definitions.KM1D import DEFAULT_EQ_PROPS, DEFAULT_MAT_PROPS, DEFAULT_SOLVER_OPTS
from gpu_numba_solver.system_definitions.KM1D import CP, DP, SP
from gpu_numba_solver import GPU_ODE
from gpu_numba_solver.GPU_ODE import SolverObject
EQ_PROPS    = DEFAULT_EQ_PROPS.copy()
MAT_PROPS   = DEFAULT_MAT_PROPS.copy()
SOLVER_OPTS = DEFAULT_SOLVER_OPTS.copy()



class Pos1B1D(BubbleGPUEnv):
    def __init__(self,
                 num_envs: int = 1, 
                 R0: float = 40.0,
                 components: int = 1,
                 ac_type: str = "SW_N",
                 freqs : List[float] = [25.0],
                 pa : List[float] = [1.0],
                 phase_shift : List[float] = [0.0], 
                 action_space_dict: ActionSpaceDict = ActionSpaceDict({"IDX": [0], "MIN": [0.0], "MAX": [1.5], "TYPE": "Box"},
                                                                      {"IDX": [0], "MIN": [0.0], "MAX": [0.25*torch.pi], "TYPE": "Box"}),
                 observation_space_dict: ObservationSpaceDict = ObservationSpaceDict(XT = {"IDX": [0], "MIN": [0.0], "MAX": [0.25], "TYPE": "Box"},
                                                                                      X = {"IDX": [0], "MIN": [0.0], "MAX": [0.25], "TYPE": "Box", "STACK": 2} ),
                 rel_freq: Optional[float] = None,
                 episode_length: int = 10,
                 time_step_length: Union[int, float] = 50,
                 seed: int = None,
                 target_position: Union[str, float] = "random",
                 initial_position: Union[str, float] = "random",
                 position_tolerace: float = 1e-2,
                 apply_termination: bool = True,
                 render_env: bool = False,
                 collect_trajectories: bool = False,
                 dense_output_resolution: int = 1000,
                 save_file_name: str = "Pos1B1D" ) -> None:

        # Update Global Dictionaries
        super().__init__(num_envs=num_envs,
                        action_space_dict=action_space_dict,
                        observation_space_dict=observation_space_dict,
                        seed=seed)
        
        
        SOLVER_OPTS["NT"] = num_envs
        SOLVER_OPTS["NDO"] = dense_output_resolution if collect_trajectories else 0
        EQ_PROPS["k"] = components
        if all(len(lst) == components for lst in [freqs, pa, phase_shift]):
            for i in range(components):
                EQ_PROPS["FREQ"][i] = freqs[i] * 1.0e3
                EQ_PROPS["PS"][i]   = phase_shift[i]
                EQ_PROPS["PA"][i]   = pa[i] * 1.0e5 
        else:
            print("Err: The number of components differs from the length of the provided parameter lists.")
            exit()
        EQ_PROPS["REL_FREQ"] = EQ_PROPS["FREQ"][0] if rel_freq == None else rel_freq * 1.0e3

        # Physical Parameters
        self.R0 = R0 * 1e-6

        self.episode_length          = episode_length
        self.time_step_length        = float(time_step_length) if isinstance(time_step_length, int) else time_step_length
        self.action_space_dict       = action_space_dict
        self.observation_space_dict  = observation_space_dict
        self.components              = components

        assert (target_position == "random" or type(target_position) in [float, int]), "Err target_position = 'random' or float number"
        self.target_position = target_position
        assert (initial_position == "random" or type(initial_position) in [float, int]), "Err target_position = 'random' or float number"
        self.initial_position = initial_position

        self.position_tolerance = position_tolerace
        self.apply_termination  = apply_termination

        self.render_env = render_env
        # Collect Radius (R - 0) and Position (X - 1)
        self.trajector_collector = TrajectorCollector(save_file_name=save_file_name, num_envs=num_envs, state_index=[0, 1]) if collect_trajectories else None

        # Configure Solver Object ---------------------------------------
        if ac_type not in ["CONST", "SW_N", "SW_A"]: 
            print("Err: Acoustic field type is not supported!")
            exit()
        GPU_ODE.setup(KM1D, k=EQ_PROPS["k"], ac_field=ac_type)
        self.num_envs = num_envs
        self.solver = SolverObject(
            number_of_threads=SOLVER_OPTS["NT"],
            system_dimension=SOLVER_OPTS["SD"],
            number_of_control_parameters=SOLVER_OPTS["NCP"],
            number_of_shared_parameters=SOLVER_OPTS["NSP"],
            number_of_dynamic_parameters=SOLVER_OPTS["NDP"]*EQ_PROPS["k"],
            number_of_accessories=SOLVER_OPTS["NACC"],
            number_of_dense_outputs=SOLVER_OPTS["NDO"],
            method=SOLVER_OPTS["SOLVER"],
            abs_tol=SOLVER_OPTS["ATOL"],
            rel_tol=SOLVER_OPTS["RTOL"]
        )

        self.single_observation_space = self.observation_space.single_space()
        self.single_action_space = self.action_space.single_space()
        
        # Collections
        self.index_map = {"R": 0, "X": 1, "U": 2, "V": 3}
        self.observed_variables = {}
        for attr in self.observation_space_dict.observed_quatities:
            obs = getattr(self.observation_space_dict, attr)
            if obs is not None:
                max_len = obs["STACK"] if "STACK" in obs.keys() else 1
                self.observed_variables[attr] = {"values": deque([], maxlen=max_len), "len": max_len}


    def reset(self, **kwargs):
        # Set Initial Positions --------------
        if (self.initial_position == "random"):
            bubble_positions = torch.rand(size=(self.num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.X["MAX"][0] - self.observation_space_dict.X["MIN"][0] ) + self.observation_space_dict.X["MIN"][0]
            
        elif type(self.initial_position) in [float, int]:
            bubble_positions = torch.full(size=(self.num_envs,), fill_value=self.initial_position, dtype=torch.float32, device="cuda").contiguous()
        else:
            print("Err: initial bubble position is not created!")


        # Set Target positions ---------------
        if (self.target_position == "random"):
            target_positions = torch.rand(size=(self.num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.XT["MAX"][0] - self.observation_space_dict.XT["MIN"][0] ) + self.observation_space_dict.XT["MIN"][0]
            
        elif type(self.target_position) in [float, int]:
            target_positions = torch.full(size=(self.num_envs, ), fill_value=self.target_position, dtype=torch.float32, device="cuda").contiguous()
        else:
            print("Err: target bubble position is not created!")

        if "XT" in self.observed_variables.keys():
            for _ in range(self.observed_variables["XT"]["len"]):
                self.observed_variables["XT"]["values"].appendleft(target_positions.clone())

    
        if "X" in self.observed_variables.keys():
            for _ in range(self.observed_variables["X"]["len"]):
                self.observed_variables["X"]["values"].appendleft(bubble_positions.clone())
        
        
        # Initialize Radius
        # TODO: add random size distribution
        self.R0s = torch.full(size=(self.num_envs, ), fill_value=self.R0, dtype=torch.float32, device="cuda").contiguous()


        # Set Host properties
        for tid in range(self.num_envs):
            self.solver.set_host(tid, "time_domain",  0, 0.0)
            self.solver.set_host(tid, "time_domain",  1, self.time_step_length)
            self.solver.set_host(tid, "actual_state", 0, 1.0)
            self.solver.set_host(tid, "actual_state", 1, bubble_positions[tid].item())
            self.solver.set_host(tid, "actual_state", 2, 0.0)
            self.solver.set_host(tid, "actual_state", 3, 0.0)

            if self.trajector_collector is not None:
                self.trajector_collector.trajectories[tid].episode_radii = [self.R0s[tid].item()]    

        self._fill_control_parameters()
        self._fill_shared_parameters()
        self._fill_dynamic_parameters()

        self.solver.syncronize_h2d("all")
        
        self._get_observation()

        self.algo_steps    = torch.zeros(size=(self.num_envs, ), dtype=torch.long, device="cuda")
        self.total_rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device="cuda")

        return self._observation, {}

    def reset_envs(self, env_ids):
        num_envs = len(env_ids)
        # Set Initial Positions --------------
        if (self.initial_position == "random"):
            bubble_positions = torch.rand(size=(num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.X["MAX"][0] - self.observation_space_dict.X["MIN"][0] ) + self.observation_space_dict.X["MIN"][0]
            
        elif type(self.initial_position) in [float, int]:
            bubble_positions = torch.full(size=(num_envs,), fill_value=self.initial_position, dtype=torch.float32, device="cuda").contiguous()
        else:
            print("Err: initial bubble position is created!")
 

        # Set Target positions ---------------
        if (self.target_position == "random"):
            target_positions = torch.rand(size=(num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.XT["MAX"][0] - self.observation_space_dict.XT["MIN"][0] ) + self.observation_space_dict.XT["MIN"][0]
            
        elif type(self.target_position) in [float, int]:
            target_positions = torch.full(size=(num_envs, ), fill_value=self.target_position, dtype=torch.float32, device="cuda").contiguous()
        else:
            print("Err: target bubble position is not created!")


        for i, tid in enumerate(env_ids):
            if "XT" in self.observed_variables.keys():
                for j in range(self.observed_variables["XT"]["len"]):
                    self.observed_variables["XT"]["values"][j][tid] = target_positions[i]

        for i, tid in enumerate(env_ids):
            if "X" in self.observed_variables.keys():
                for j in range(self.observed_variables["X"]["len"]):
                    self.observed_variables["X"]["values"][j][tid] = bubble_positions[i]

        # Initialize Radius
        # TODO: add random size distribution
        #self.R0s = torch.full(size=(self.num_envs, ), fill_value=self.R0, dtype=torch.float32, device="cuda").contiguous()

        # Reset the solver
        for tid in env_ids:
            self.solver.set_device(tid, "time_domain",  0, 0.0)
            self.solver.set_device(tid, "time_domain",  1, self.time_step_length)
            self.solver.set_device(tid, "actual_state", 0, 1.0)
            self.solver.set_device(tid, "actual_state", 1, self.bubble_positions()[tid].item())
            self.solver.set_device(tid, "actual_state", 2, 0.0)
            self.solver.set_device(tid, "actual_state", 3, 0.0)

            if self.trajector_collector is not None:
                self.trajector_collector.trajectories[tid].episode_radii = [self.R0s[tid].item()]

            self.algo_steps[tid] = 0
            self.total_rewards[tid] = 0.0

        self.solver.syncronize()
        self._get_observation()

    def step(self, action: torch.Tensor = None):
        
        # ----- Update action -----
        if action == None:
            action = self.action_space.sample()
        self._set_action_on_environment(action)

        # ---- Run the solver ------
        self.solver.solve_my_ivp()
        self.algo_steps    += 1

        # --- Get New Observation ---
        # 1) update the deques containing the observed values
        # 2) create new observation tensonr
        #print("step before done")
        self._observe_environment()
        self._get_observation()
 
        # --- Check the terminal states ----
        self._termination_and_truncation()

        # ---- Calculate the reward ------
        self._get_rewards()
        self.total_rewards += self._rewards

        # --- Render Env ----
        if self.render_env:
            self.render()

        # ---- Collect Trajectories ----
        if self.trajector_collector is not None:
            self.solver.syncronize_d2h("dense_output")
            self.solver.syncronize()
            dense_index, dense_time, dense_states = self.solver.get_dense_output()
            self.trajector_collector.step(self._observation.cpu().numpy().astype(np.float32),
                                          self._actions.cpu().numpy().astype(np.float32),
                                          self._rewards.cpu().numpy().astype(np.float32),
                                          dense_index,
                                          dense_time,
                                          dense_states)

        # --- Handle final observation 
        info = {}
        #print(self._time_out)
        #print(self._positive_terminal)
        #print(self._negative_terminal)
        #print(self.observed_variables)
        #print(self._observation)
        done_env_idx = torch.nonzero(self._time_out + self._positive_terminal + self._negative_terminal).flatten()
        if len(done_env_idx) > 0:
            info = {
                "final_observation" : self._observation[done_env_idx].clone(),
                "dones"             : done_env_idx,
                "episode_return"    : self.total_rewards[done_env_idx].clone(),
                "episode_length"    : self.algo_steps[done_env_idx].clone()
            }

            if self.trajector_collector is not None:
                self.trajector_collector.end_episode(done_env_idx,
                                                     self._observation[done_env_idx].cpu().numpy().astype(np.float32),
                                                     self.algo_steps[done_env_idx].cpu().numpy().astype(np.int32),
                                                     self.total_rewards[done_env_idx].cpu().numpy().astype(np.float32))

            self.reset_envs(done_env_idx)

        return (self._observation.clone(),
                self._rewards.clone(),
                (self._positive_terminal + self._negative_terminal).clone(),
                self._time_out.clone(),
                info)

              
    def render(self):
        try:
            ax0 = self.fig.axes[0]
            ax1 = self.fig.axes[1]
            ax2 = self.fig.axes[2]
            ax0.cla()
            ax1.cla()
            ax2.cla()
        except:
            self.fig = plt.figure(1, figsize=(6, 8))
            ax0 = self.fig.add_subplot(3, 1, 1)
            ax1 = self.fig.add_subplot(3, 1, 2)
            ax2 = self.fig.add_subplot(3, 1, 3)
        finally:
            ax0.plot([id for id in range(self.num_envs)], self.target_positions().cpu().numpy(), "g+", markersize=10)
            ax0.plot([id for id in range(self.num_envs)], self.bubble_positions().cpu().numpy(), "k.", markersize=10)
            #ax0.plot([id for id in range(self.num_envs)], position1, "r.", markersize=5)
            ax0.set_ylim(-0.05, 0.35)     # TODO: 
            ax0.set_xlabel(r"Environment")
            ax0.set_ylabel(r"$x/\lambda_r$")
            ax0.grid("both")


            ax1.plot([id for id in range(self.num_envs)], self._actions.cpu().numpy(), ".", markersize=5)
            ax1.set_ylim(0, 2)

            ax2.plot([id for id in range(self.num_envs)], self._rewards.cpu().numpy(), "k.", markersize=5)

            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    # TODO: Create propery!!
    def target_positions(self):
        return self.observed_variables["XT"]["values"][0]

    def bubble_positions(self):
        return self.observed_variables["X"]["values"][0]


    # ------ Environment's private methods ------------
    def _get_rewards(self):
        max_distance = torch.max(
            torch.hstack((
                (self.observation_space_dict.X["MAX"][0] - self.target_positions()).unsqueeze(-1),
                (self.target_positions() - self.observation_space_dict.X["MIN"][0]).unsqueeze(-1))), dim=1)[0]
        
        target_distance = abs(self.bubble_positions() - self.target_positions())

        self._rewards = - (target_distance / max_distance)**0.5   \
                       -10 * self._negative_terminal  \
                       +10 * self._positive_terminal

    def _termination_and_truncation(self):
        self._negative_terminal = (torch.tensor(self.solver.status(), device="cuda") != 0).squeeze()      # Ode solver failure

        if self.apply_termination:
            self._positive_terminal = abs(self.bubble_positions() - self.target_positions()) < self.position_tolerance

        self._time_out = (self.algo_steps == self.episode_length)

    def _get_observation(self):
        observation = []
        for key in self.observed_variables.keys():
            obs = self.observed_variables[key]
            for i in range(obs["len"]):
                observation.append(obs["values"][i].unsqueeze(1))

        self._observation = self.observation_space.normalize(torch.hstack(observation))

    # ------ Solver Pytorch API Interface -----
    def _observe_environment(self):
        for key in self.observed_variables.keys():
            if key in self.index_map:
                obs = self.observed_variables[key]
                obs["values"].appendleft(torch.tensor(self.solver.get_device_array("actual_state", self.index_map[key]), device="cuda", dtype=torch.float32).clone())


    def _set_action_on_environment(self, action: torch.Tensor):
        self._actions = action
        shift = 0
        if self.action_space_dict.PA is not None:
            for idx in self.action_space_dict.PA["IDX"]:
                self.solver.set_device_array("dynamic_parameters", idx, action[:,idx].contiguous().to(dtype=torch.float64) * 1.0e5)
                shift +=1
        if self.action_space_dict.PS is not None:
            k = self.components*2
            for idx in self.action_space_dict.PS["IDX"]:
                self.solver.set_device_array("dynamic_parameters", idx+k, action[:,shift+idx].contiguous().to(dtype=torch.float64) * torch.pi)
        if self.action_space_dict.FR is not None:
            pass

    def _fill_control_parameters(self):
        # Equation properties
        for tid in range(self.num_envs):
            EQ_PROPS["R0"] = self.R0s[tid].item()
            for (k, f) in CP.items():
                self.solver.set_host(tid, "control_parameters", k, f(**MAT_PROPS, **EQ_PROPS))

    def _fill_shared_parameters(self):
        for (k, f) in SP.items():
            self.solver.set_shared_host("shared_parameters", k, f(**MAT_PROPS, **EQ_PROPS))

    def _fill_dynamic_parameters(self):
        # Acoustic field properties
        for tid in range(self.num_envs):
            for (k, f) in DP.items():
                for i in range(EQ_PROPS["k"]):
                    self.solver.set_host(tid, "dynamic_parameters", i + k*EQ_PROPS["k"], f(i, **MAT_PROPS, **EQ_PROPS))
