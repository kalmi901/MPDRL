import torch
from numba import cuda
from matplotlib import pyplot as plt
from typing import List, Optional, Union

from envs import BubbleGPUEnv
from envs import ActionSpaceDict

from envs.common import VSpace
from envs.common import SpaceType
from envs.common import Box
from envs.common import Discrete
from envs.common import Hybrid

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
                 rel_freq: Optional[float] = None,
                 stacked_frames: int = 2,
                 bubble_posistion_limits: tuple = (0.0, 0.25),
                 target_position_limits: tuple = (0.0, 0.25),
                 episode_length: int = 10,
                 time_step_length: Union[int, float] = 50,
                 seed: int = None,) -> None:

        # Update Global Dictionaries
        super().__init__(num_envs=num_envs,
                        action_space_dict=action_space_dict,
                        seed=seed)
        
        
        SOLVER_OPTS["NT"] = num_envs
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
        self.R0 = torch.full((num_envs, 1), R0 * 1e-6, dtype=torch.float64)     # Fix for all bubbles (size dependece must be added to the observation)

        self.bubble_posistion_limits = bubble_posistion_limits
        self.target_position_limits  = target_position_limits
        self.episode_length          = episode_length
        self.time_step_length        = float(time_step_length) if isinstance(time_step_length, int) else time_step_length
        self.action_space_dict       = action_space_dict
        self.components              = components

        # Configure Solver Object ---------------------------------------
        if ac_type not in ["CONST", "SW_N", "SW_A"]: 
            print("Err: Acoustic field type is not supported!")
        GPU_ODE.setup(KM1D, k=EQ_PROPS["k"], ac_field=ac_type)
        self.num_envs = num_envs
        self.solver = SolverObject(
            number_of_threads=SOLVER_OPTS["NT"],
            system_dimension=SOLVER_OPTS["SD"],
            number_of_control_parameters=SOLVER_OPTS["NCP"],
            number_of_shared_parameters=SOLVER_OPTS["NSP"],
            number_of_dynamic_parameters=SOLVER_OPTS["NDP"]*EQ_PROPS["k"],
            number_of_accessories=SOLVER_OPTS["NACC"],
            method=SOLVER_OPTS["SOLVER"],
            abs_tol=SOLVER_OPTS["ATOL"],
            rel_tol=SOLVER_OPTS["RTOL"]
        )

        # Configure Environment ----------------------------------------
        self.observation_space = Box(low=torch.Tensor([target_position_limits[0]] + [bubble_posistion_limits[0] for _ in range(stacked_frames)]),
                                     high=torch.Tensor([target_position_limits[1]] + [bubble_posistion_limits[1] for _ in range(stacked_frames)]),
                                     num_envs=self.num_envs,
                                     dtype=torch.float64,
                                     seed=seed)
        

        self.single_observation_space = self.observation_space.single_space()
        self.single_action_space = self.action_space.single_space()

        self.single_observation_space_shape = self.observation_space.single_shape
        self.single_action_space_shape      = self.action_space.single_shape
        self.single_action_space_low        = self.action_space.low
        self.single_action_space_high       = self.action_space.high



    def reset(self, **options):
        for tid in range(self.num_envs):
            self.solver.set_host(tid, "time_domain",  0, 0.0)
            self.solver.set_host(tid, "time_domain",  1, self.time_step_length)
            self.solver.set_host(tid, "actual_state", 0, 1.0)
            self.solver.set_host(tid, "actual_state", 1, 0.0)
            self.solver.set_host(tid, "actual_state", 2, 0.0)
            self.solver.set_host(tid, "actual_state", 3, 0.0)

        self._fill_control_parameters()
        self._fill_shared_parameters()
        self._fill_dynamic_parameters()
        self.solver.syncronize_h2d("all")
        
        self.actual_observation = self.observation_space.sample()
        self.actual_action      = self.action_space.sample()
        # Copy positions to device
        self.solver.set_device_array("actual_state", 1, self.actual_observation[:,1].contiguous())

        return self.actual_observation, {}
    

    def render(self):
        target   = self.actual_observation[:,0].detach().cpu().numpy()
        position = self.actual_observation[:,1].detach().cpu().numpy()

        try:
            ax0 = self.fig.axes[0]
            ax0.cla()
        except:
            self.fig = plt.figure(1, figsize=(12, 7))
            ax0 = self.fig.add_subplot(1, 1, 1)
        finally:
            ax0.plot([id for id in range(self.num_envs)], target,   "g.", markersize=10)
            ax0.plot([id for id in range(self.num_envs)], position, "b.", markersize=10)
            ax0.set_ylim(min(self.bubble_posistion_limits)-0.015, max(self.bubble_posistion_limits)+0.015)
            ax0.set_xlabel(r"Environment")
            ax0.set_ylabel(r"$x/\lambda_r$")
            ax0.grid("both")

            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    def _reset_subenvs(self, dones):
        self.actual_observation[dones] = self.observation_space.sample()[dones]
        self.total_reward[dones]    *=0
        self.steps_done[dones]      *= 0

        for tid in dones:
            self.solver.set_device(tid, "time_domain",  0, 0.0)
            self.solver.set_device(tid, "time_domain",  1, self.time_step_length)
            self.solver.set_device(tid, "actual_state", 0, 1.0)
            self.solver.set_device(tid, "actual_state", 0, 1.0)
            self.solver.set_device(tid, "actual_state", 1, 0.0)
            self.solver.set_device(tid, "actual_state", 2, 0.0)
            self.solver.set_device(tid, "actual_state", 3, 0.0)

            """ If R0 changes between episodes
            EQ_PROPS["R0"] = self.R0[tid].item()
            for (k, f) in CP.items():
                self.solver.set_device(tid, "control_parameters", k, f(**MAT_PROPS, **EQ_PROPS))
            """

        self.solver.syncronize()
        self.solver.set_device_array("actual_state", 1, self.actual_observation[:,1].contiguous())

    def _final_observation_and_reset(self):
        dones = torch.nonzero(torch.logical_or(self.actual_terminated, self.actual_time_out) == True).squeeze()

        if len(dones) > 0:
            self.actual_info ={
            "final_observation" : self.actual_observation[dones],
            "dones"             : dones,
            "episode_return"    : self.total_reward[dones],
            "episode_length"    : self.steps_done[dones]
            }

            self._reset_subenvs(dones)

    def _get_terminal_and_rewards(self):
        # Check for Integration Failures
        fail_idx = torch.nonzero(torch.tensor(self.solver.status()) != 0).squeeze()
        self.actual_terminated[fail_idx] = True

        # Check for timeouts
        time_out_idx = torch.nonzero(self.steps_done == self.episode_length).squeeze()
        self.actual_time_out[time_out_idx] = True

        self.actual_reward = 1 - (self.actual_observation[:, 0] - self.actual_observation[:, 1]) / 0.25

        # Penalize for failure!
        self.actual_reward[fail_idx] = - 100

        self.total_reward += self.actual_reward        

    
    def _get_observations(self):
        self.actual_observation[:, 2] = self.actual_observation[:, 1]
        self.actual_observation[:, 1] = torch.tensor(self.solver.get_device_array("actual_state", 1))
    
    def _set_action(self):
        shift = 0
        if self.action_space_dict.PA is not None:
            for idx in self.action_space_dict.PA["IDX"]:
                self.solver.set_device_array("dynamic_parameters", idx, self.actual_action[:,idx].contiguous().to(dtype=torch.float64) * 1.0e5)
                shift +=1
        if self.action_space_dict.PS is not None:
            k = self.components*2
            for idx in self.action_space_dict.PS["IDX"]:
                self.solver.set_device_array("dynamic_parameters", idx+k, self.actual_action[:,shift+idx].contiguous().to(dtype=torch.float64))
        if self.action_space_dict.FR is not None:
            pass

    def _fill_control_parameters(self):
        # Equation properties
        for tid in range(self.num_envs):
            EQ_PROPS["R0"] = self.R0[tid].item()
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