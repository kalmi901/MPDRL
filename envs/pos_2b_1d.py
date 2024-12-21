import torch
from collections import deque
from matplotlib import pyplot as plt
from typing import List, Optional, Union

from envs import BubbleGPUEnv
from envs import ActionSpaceDict
from envs import ObservationSpaceDict

# SOLVER SPECIFIC PARAMETERS
# System definition
from gpu_numba_solver.system_definitions import KM1D2B
from gpu_numba_solver.system_definitions.KM1D2B import DEFAULT_EQ_PROPS, DEFAULT_MAT_PROPS, DEFAULT_SOLVER_OPTS
from gpu_numba_solver.system_definitions.KM1D2B import CP, DP, SP
from gpu_numba_solver import GPU_ODE
from gpu_numba_solver.GPU_ODE import SolverObject
EQ_PROPS    = DEFAULT_EQ_PROPS.copy()
MAT_PROPS   = DEFAULT_MAT_PROPS.copy()
SOLVER_OPTS = DEFAULT_SOLVER_OPTS.copy()


class Pos2B1D(BubbleGPUEnv):
    def __init__(self, 
                 num_envs: int = 1,
                 R0: List[float] = [60.0, 60.0],
                 components: int = 1,
                 ac_type: str = "SW_N",
                 freqs: List[float] = [25.0],
                 pa: List[float] = [1.0],
                 phase_shift : List[float] = [0.0], 
                 action_space_dict: ActionSpaceDict = ActionSpaceDict({"IDX": [0], "MIN": [0.0], "MAX": [1.5], "TYPE": "Box"},
                                                                      {"IDX": [0], "MIN": [0.0], "MAX": [0.25*torch.pi], "TYPE": "Box"}), 
                 observation_space_dict: ObservationSpaceDict = ObservationSpaceDict(XT = {"IDX": [0], "MIN": [0.0], "MAX": [0.25], "TYPE": "Box"},
                                                                                      X = {"IDX": [0, 1], "MIN": [-0.5, -0.5], "MAX": [0.5, 0.5], "TYPE": "Box", "STACK": 2} ),
                 rel_freq: Optional[float] = None,
                 episode_length: int = 100,
                 time_step_length: Union[int, float] = 10,
                 seed: int = None,
                 target_position: Union[str, float] = "random",
                 initial_position: Union[str, float] = "random",
                 min_distance: float = 0.1,
                 max_distance: float = 0.25,
                 position_tolerance: float = 0.01,
                 apply_termination: bool = True,
                 reward_weight: List[float] = [0.5, 0.25, 0.25],
                 reward_exp: float = 0.3,
                 positive_terminal_reward: float = 100.0,
                 negative_terminal_reward: float = -100.0,
                 render_env: bool = False) -> None:
        
        # Update Global Dictionaries
        super().__init__(num_envs, action_space_dict, observation_space_dict, seed)

        SOLVER_OPTS["NT"] = num_envs
        EQ_PROPS["k"] = components
        EQ_PROPS["FREQ"] = []
        EQ_PROPS["PS"] = []
        EQ_PROPS["PA"] = []
        if all(len(lst) == components for lst in [freqs, pa, phase_shift]):
            for i in range(components):
                EQ_PROPS["FREQ"].append(freqs[i] * 1.0e3)
                EQ_PROPS["PS"].append(phase_shift[i])
                EQ_PROPS["PA"].append(pa[i] * 1.0e5)
        else:
            print("Err: The number of components differs from the length of the provided parameter lists.")
            exit()
        EQ_PROPS["REL_FREQ"] = EQ_PROPS["FREQ"][0] if rel_freq == None else rel_freq * 1.0e3

        # Physical Parameters
        self.R0                     = [r * 1e-6 for r in R0]

        self.episode_length          = episode_length
        self.time_step_length        = float(time_step_length) if isinstance(time_step_length, int) else time_step_length
        self.action_space_dict       = action_space_dict
        self.observation_space_dict  = observation_space_dict
        self.components              = components

        assert (target_position == "random" or type(target_position) in [float, int]), "Err target_position = 'random' or float number"
        self.target_position = target_position
        assert (initial_position == "random" or type(initial_position) in [float, int]), "Err target_position = 'random' or float number"
        self.initial_position = initial_position

        self.position_tolerance = position_tolerance
        self.apply_termination  = apply_termination

        self.min_distance = min_distance
        self.max_distance = max_distance

        self.w = torch.tensor(reward_weight, dtype=torch.float32, requires_grad=False)
        self.b = reward_exp
        self.positive_terminal_reward = positive_terminal_reward
        self.negative_terminal_reward = negative_terminal_reward

        self.render_env = render_env

        # Configure Solver Object
        if ac_type not in ["CONST", "SW_N", "SW_A"]: 
            print("Err: Acoustic field type is not supported!")
            exit()
        GPU_ODE.setup(KM1D2B, k=EQ_PROPS['k'], ac_field=ac_type)
        self.num_envs = num_envs
        self.solver = SolverObject(
            number_of_threads=SOLVER_OPTS["NT"],
            system_dimension=SOLVER_OPTS["SD"],
            number_of_control_parameters=SOLVER_OPTS["NCP"],
            number_of_shared_parameters=SOLVER_OPTS["NSP"],
            number_of_dynamic_parameters=SOLVER_OPTS["NDP"] * EQ_PROPS["k"],
            number_of_accessories=SOLVER_OPTS["NACC"],
            number_of_events=SOLVER_OPTS["NE"],
            method=SOLVER_OPTS["SOLVER"],
            abs_tol=SOLVER_OPTS["ATOL"],
            rel_tol=SOLVER_OPTS["RTOL"],
            event_tol=SOLVER_OPTS["ETOL"],
            event_dir=SOLVER_OPTS["EDIR"]
        )

        self.single_observation_space = self.observation_space.single_space()
        self.single_action_space = self.action_space.single_space()

        # Collections
        self.index_map = {"R_1": 0, "R_2": 1, "X_0": 2, "X_1": 3, "U_0": 4, "U_1": 5, "V_0": 6, "V_1": 7}
        self.observed_variables = {}
        for attr in self.observation_space_dict.observed_quatities:
            obs = getattr(self.observation_space_dict, attr)
            if obs is not None:
                max_len =  obs["STACK"] if "STACK" in obs.keys() else 1
                for bub_idx in obs["IDX"]:
                    if len(obs["IDX"]) == 1:
                        self.observed_variables[attr] = {"values": deque([], maxlen=max_len), "len": max_len}
                    else:
                        self.observed_variables[attr+f'_{bub_idx:.0f}'] = {"values": deque([], maxlen=max_len), "len": max_len}


    def reset(self, **kwargs):
        if (self.initial_position == "random"):
            mean_postion = torch.rand(size=(self.num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.X["MAX"][0] - self.observation_space_dict.X["MIN"][0] ) + self.observation_space_dict.X["MIN"][0]

        elif type(self.initial_position) in [float, int]:
            mean_postion  = torch.full(size=(self.num_envs,), fill_value=self.initial_position, dtype=torch.float32, device="cuda").contiguous()
            
        else:
            print("Err: bubble positions are not initialized!")

        distance = torch.rand(size=(self.num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                    * (2*self.max_distance - self.min_distance) + self.min_distance

        bubble_pos_0 = torch.clamp((mean_postion - 0.5*distance), self.observation_space_dict.X["MIN"][0], self.observation_space_dict.X["MAX"][0])
        bubble_pos_1 = torch.clamp((bubble_pos_0 + 1.0*distance), self.observation_space_dict.X["MIN"][1], self.observation_space_dict.X["MAX"][1])

        if (self.target_position == "random"):
            target_positions = torch.rand(size=(self.num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.XT["MAX"][0] - self.observation_space_dict.XT["MIN"][0] ) + self.observation_space_dict.XT["MIN"][0]    

        elif type(self.target_position) in [float, int]:
            target_positions = torch.full(size=(self.num_envs, ), fill_value=self.target_position, dtype=torch.float32, device="cuda").contiguous()
        
        else:
            print("Err: target position is not initialized!")


        # ----  Initialize first observation!
        if "XT" in self.observed_variables.keys():
            for _ in range(self.observed_variables["XT"]["len"]):
                self.observed_variables["XT"]["values"].appendleft(target_positions.clone())


        if "X_0" in self.observed_variables.keys():
            for _ in range(self.observed_variables["X_0"]["len"]):
                self.observed_variables["X_0"]["values"].appendleft(bubble_pos_0.clone())

        if "X_1" in self.observed_variables.keys():
            for _ in range(self.observed_variables["X_1"]["len"]):
                self.observed_variables["X_1"]["values"].appendleft(bubble_pos_1.clone())

        # TODO: support radius!


        # TODO: randomize bubble positions!
        # self.R0 

        # ----- Initialize ODE solver Initial Conditions
        # Set time domain and actual state
        for tid in range(self.num_envs):
            self.solver.set_host(tid, "time_domain", 0, 0.0)
            self.solver.set_host(tid, "time_domain", 1, self.time_step_length)
            self.solver.set_host(tid, "actual_state", 0, 1.0)   # R0_0
            self.solver.set_host(tid, "actual_state", 1, 1.0)   # R0_1
            self.solver.set_host(tid, "actual_state", 2, bubble_pos_0[tid])     # x0_0
            self.solver.set_host(tid, "actual_state", 3, bubble_pos_1[tid])     # x0_1
            self.solver.set_host(tid, "actual_state", 4, 0.0)
            self.solver.set_host(tid, "actual_state", 5, 0.0)
            self.solver.set_host(tid, "actual_state", 6, 0.0)
            self.solver.set_host(tid, "actual_state", 7, 0.0)


        self._fill_control_parameters()
        self._fill_shared_parameters()
        self._fill_dynamic_parameters()
                
        self.solver.syncronize_h2d("all")

        self.algo_steps = torch.zeros(size=(self.num_envs, ), dtype=torch.long, device="cuda")
        self.total_rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device="cuda")


        self._get_observation()
        return self._observation, {}


    def reset_envs(self, env_ids):
        num_envs = len(env_ids)
        if (self.initial_position == "random"):
            mean_postion = torch.rand(size=(num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.X["MAX"][0] - self.observation_space_dict.X["MIN"][0] ) + self.observation_space_dict.X["MIN"][0]

        elif type(self.initial_position) in [float, int]:
            mean_postion  = torch.full(size=(num_envs,), fill_value=self.initial_position, dtype=torch.float32, device="cuda").contiguous()
            
        else:
            print("Err: bubble positions are not initialized!")

        distance = torch.rand(size=(num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                    * (2*self.max_distance - self.min_distance) + self.min_distance

        bubble_pos_0 = torch.clamp((mean_postion - 0.5*distance), self.observation_space_dict.X["MIN"][0], self.observation_space_dict.X["MAX"][0])
        bubble_pos_1 = torch.clamp((bubble_pos_0 + 1.0*distance), self.observation_space_dict.X["MIN"][1], self.observation_space_dict.X["MAX"][1])

        if (self.target_position == "random"):
            target_positions = torch.rand(size=(num_envs, ), dtype=torch.float32, device="cuda").contiguous() \
                * (self.observation_space_dict.XT["MAX"][0] - self.observation_space_dict.XT["MIN"][0] ) + self.observation_space_dict.XT["MIN"][0]    

        elif type(self.target_position) in [float, int]:
            target_positions = torch.full(size=(num_envs, ), fill_value=self.target_position, dtype=torch.float32, device="cuda").contiguous()
        
        else:
            print("Err: target position is not initialized!")

        for i, tid in enumerate(env_ids):
            if "XT" in self.observed_variables.keys():
                for j in range(self.observed_variables["XT"]["len"]):
                    self.observed_variables["XT"]["values"][j][tid] = target_positions[i]

            if "X_0" in self.observed_variables.keys():
                for j in range(self.observed_variables["X_0"]["len"]):
                    self.observed_variables["X_0"]["values"][j][tid] = bubble_pos_0[i]

            if "X_1" in self.observed_variables.keys():
                for j in range(self.observed_variables["X_1"]["len"]):
                    self.observed_variables["X_1"]["values"][j][tid] = bubble_pos_1[i]

            # TODO: Add Radius

            # Reset the solver
        for i, tid in enumerate(env_ids):
            self.solver.set_device(tid, "time_domain", 0, 0.0)
            self.solver.set_device(tid, "time_domain", 1, self.time_step_length)
            self.solver.set_device(tid, "actual_state", 0, 1.0)
            self.solver.set_device(tid, "actual_state", 1, 1.0)
            self.solver.set_device(tid, "actual_state", 2, bubble_pos_0[i])
            self.solver.set_device(tid, "actual_state", 3, bubble_pos_1[i])
            self.solver.set_device(tid, "actual_state", 4, 0.0)
            self.solver.set_device(tid, "actual_state", 5, 0.0)
            self.solver.set_device(tid, "actual_state", 6, 0.0)
            self.solver.set_device(tid, "actual_state", 7, 0.0)

            self.algo_steps[tid] = 0
            self.total_rewards[tid] = 0.0

        self.solver.syncronize()
        self._get_observation()


    def step(self, action: torch.Tensor = None):

        # ------- Update action --------
        if action == None:
            action = self.action_space.sample()
        self._set_action_on_environment(action)

        # ----- Run the Solver  --------
        self.solver.solve_my_ivp()
        self.algo_steps += 1

        # -- Get New Observations ---
        # 1) update the deques containing the observed values
        # 2) create new observation tensor
        self._observe_environment()
        self._get_observation()


        # --- Check the terminal states ---
        self._termination_and_truncation()

        # --- Calculate the reward -------
        self._get_rewards()
        self.total_rewards += self._rewards

        # --- Render Env ----
        if self.render_env:
            self.render()

        # -- Handle final observation
        info = {}
        done_env_idx = torch.nonzero(self._time_out + self._positive_terminal + self._negative_terminal).flatten()
        if len(done_env_idx) > 0:
            info = {
                "final_observation" : self._observation[done_env_idx].clone(),
                "dones"             : done_env_idx,
                "episode_return"    : self.total_rewards[done_env_idx].clone(),
                "episode_length"    : self.algo_steps[done_env_idx].clone()
            }

            self.reset_envs(done_env_idx)

        return (self._observation.clone(),
                self._rewards.clone(),
                (self._positive_terminal + self._negative_terminal).clone(),
                self._time_out.clone(),
                info)


    def render(self):
        target = self.observed_variables["XT"]["values"][0].detach().cpu().numpy()
        bubble_pos0 = self.observed_variables["X_0"]["values"][0].detach().cpu().numpy()
        bubble_pos1 = self.observed_variables["X_1"]["values"][0].detach().cpu().numpy()
        mean_pos = 0.5 * (bubble_pos0 + bubble_pos1)

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
            ax0.plot([id for id in range(self.num_envs)], target,   "g+", markersize=4)
            ax0.plot([id for id in range(self.num_envs)], mean_pos,   "r+", markersize=4)
            ax0.plot([id for id in range(self.num_envs)], bubble_pos0, "k.", markersize=10)
            ax0.plot([id for id in range(self.num_envs)], bubble_pos1, "k.", markersize=10)
            ax0.set_ylim(-0.26, 0.26)     # TODO: 
            ax0.set_xlabel(r"Environment")
            ax0.set_ylabel(r"$x/\lambda_r$")
            ax0.grid("both")


            ax1.plot([id for id in range(self.num_envs)], self._actions.cpu().numpy(), ".", markersize=5)
            ax1.set_ylim(0, 2)

            ax2.plot([id for id in range(self.num_envs)], self._rewards.cpu().numpy(), ".", markersize=5)

            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)


    def target_positions(self):
        return self.observed_variables["XT"]["values"][0]

    def mean_positions(self):
        return 0.5 * (self.observed_variables["X_1"]["values"][0] + self.observed_variables["X_0"]["values"][0])

    def distance(self):
        return torch.abs(self.observed_variables["X_1"]["values"][0] - self.observed_variables["X_0"]["values"][0])


    # ------------- Environment's private methods -----------------
    def _get_rewards(self):
        
        max_distance    = self.observation_space_dict.XT["MAX"][0] - self.observation_space_dict.XT["MIN"][0] 
        target_distance = torch.abs(self.mean_positions() - self.target_positions())
        bubble_distance = self.distance()
        pa_idx = self.action_space_dict.PA["IDX"]
        intesity = torch.sum(self._actions[:, pa_idx], axis=1)**0.5

        self._rewards = 1.0 - self.w[0] * (target_distance / max_distance)**self.b   \
                        - self.w[1] * (  torch.max(torch.zeros_like(bubble_distance), self.min_distance - bubble_distance)
                                       + torch.max(torch.zeros_like(bubble_distance), bubble_distance - self.max_distance) ) / self.max_distance \
                        - self.w[2] * intesity \
                        +self.negative_terminal_reward * self._negative_terminal \
                        +self.positive_terminal_reward * self._positive_terminal


        

    def _termination_and_truncation(self):
        self._negative_terminal = (torch.tensor(self.solver.status(), device="cuda") != 0).squeeze() 

        if self.apply_termination:
            self._positive_terminal = abs(self.mean_positions() - self.target_positions()) < self.position_tolerance
        else:
            self._positive_terminal = torch.full_like(self._negative_terminal, False)

        self._time_out = (self.algo_steps == self.episode_length)

    def _get_observation(self):
        observation = []
        for key in self.observed_variables.keys():
            obs = self.observed_variables[key]
            for i in range(obs["len"]):
                observation.append(obs["values"][i].unsqueeze(1))

        self._observation = self.observation_space.normalize(torch.hstack(observation))

    # ------------- Solver - ENV Interface -----------------------
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
                self.solver.set_device_array("dynamic_parameters", idx, action[:, idx].contiguous().to(dtype=torch.float64) * 1.0e5)
                shift +=1
        if self.action_space_dict.PS is not None:
            k = self.components*2
            for idx in self.action_space_dict.PS["IDX"]:
                self.solver.set_device_array("dynamic_parameters", idx+k, action[:, shift+idx].contiguous().to(dtype=torch.float64))
        if self.action_space_dict.FR is not None:
            pass

    def _fill_control_parameters(self):
        for tid in range(self.num_envs):
            EQ_PROPS["R0"] = self.R0
            for (k, f) in CP.items():
                for i in range(2):
                    self.solver.set_host(tid, "control_parameters", i * SOLVER_OPTS["NCP"] // 2 + k, f(i, **MAT_PROPS, **EQ_PROPS))

    def _fill_shared_parameters(self):
        for (k, f) in SP.items():
            self.solver.set_shared_host("shared_parameters", k, f(**MAT_PROPS, **EQ_PROPS))

    def _fill_dynamic_parameters(self):
        for tid in range(self.num_envs):
            for (k, f) in DP.items():
                for i in range(EQ_PROPS["k"]):
                    self.solver.set_host(tid, "dynamic_parameters", i + k*EQ_PROPS["k"], f(i, **MAT_PROPS, **EQ_PROPS))