import torch
from numba import cuda
from matplotlib import pyplot as plt
from typing import List, Optional, Union

from envs import BubbleGPUEnv

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
                 R0: float = 50.0,
                 components: int = 2,
                 freqs : List[float] = [25.0, 50.0],
                 rel_freq: Optional[float] = None,
                 phase_shift : List[float] = [0.0, 0.0], 
                 stacked_frames: int = 2,
                 bubble_posistion_limits: tuple = (0.0, 0.25),
                 target_position_limits: tuple = (0.0, 0.25),
                 episode_length: int = 20,
                 time_step_length: Union[int, float] = 50,
                 seed: int = None,) -> None:

        # Update Global Dictionaries
        SOLVER_OPTS["NT"] = num_envs
        EQ_PROPS["k"] = components


        # Physical Parameters
        self.R0 = torch.full((num_envs, 1), R0 * 1e-6, dtype=torch.float64)     # Fix for all bubbles (size dependece must be added to the observation)

        self.bubble_posistion_limits = bubble_posistion_limits
        self.target_position_limits  = target_position_limits
        self.episode_length          = episode_length
        self.time_step_length        = float(time_step_length) if isinstance(time_step_length, int) else time_step_length
        
        # Configure Solver Object ---------------------------------------
        GPU_ODE.setup(KM1D, k=components, ac_field="CONST")
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
        
        
        #super().__init__()


        #super().__init__(num_envs, single_action_space, single_observation_space)



    def step(self, action: torch.Tensor=None):
        self.solver.solve_my_ivp()
        self._get_observations()
        print("Step: Done")



    def reset(self, **options):
        for tid in range(self.num_envs):
            self.solver.set_host(tid, "time_domain",  0, 0.0)
            self.solver.set_host(tid, "time_domain",  1, self.time_step_length)
            self.solver.set_host(tid, "actual_state", 0, 1.0)
            self.solver.set_host(tid, "actual_state", 1, 99.0)
            self.solver.set_host(tid, "actual_state", 2, 0.0)
            self.solver.set_host(tid, "actual_state", 3, 0.0)

        self._fill_control_parameters()
        self._fill_shared_parameters()
        self._fill_dynamic_parameters()
        self.solver.syncronize_h2d("all")
        
        self.actual_observation = self.observation_space.sample()
        print("from reset:")
        print(self.actual_observation)
        # Copy positions to device
        # TODO: Add to GPU ode to avoid mixing functionalities
        self.solver._d_actual_state[self.num_envs:2*self.num_envs].copy_to_device(cuda.as_cuda_array(self.actual_observation[:,1].contiguous()))
    

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



    def _get_rewards(self):
        return super()._get_rewards()
    
    def _get_observations(self):
        self.actual_observation[:, 2] = self.actual_observation[:, 1]
        self.actual_observation[:, 1] = torch.tensor(self.solver._d_actual_state[self.num_envs:2*self.num_envs])
    
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