import torch
from envs.common import Space
from envs import BubbleGPUEnv
import torch



# SOLVER SPECIFIC PARAMETERS
from gpu_numba_solver.GPU_ODE import SolverObject
SYSTEM_DEFINITION = "KM1D"
NT   = 1     # Number of Threads (min 1 active thread is required)
SD   = 6     # System dimension
NCP  = 27    # Number of control parameters
NACC = 5     # Number of accessories
SOLVER  = "RKCK45"
BLOCKSIZE = 64
ATOL = 1e-9
RTOL = 1e-9
DT_MIN = 1e-10  # Minimum TimeStep 


# MATERIAL PROPERTIES






class Pos1DControl(BubbleGPUEnv):
    def __init__(self,
                 num_envs: int, 
                 single_action_space: Space = None, 
                 single_observation_space: Space = None) -> None:
        super().__init__(num_envs, single_action_space, single_observation_space)

        NT = num_envs

        self.model_impl = SolverObject(
            number_of_threads=NT,
            system_dimension=SD,
            number_of_control_parameters=NCP,
            number_of_accessories=NACC,
            method=SOLVER,
            abs_tol=ATOL,
            rel_tol=RTOL
        )

        self.model_impl.time_step_min = DT_MIN


    def step(self, action: torch.Tensor):
        pass

    def reset(self, seed: int, **options):
        return super().reset(seed, **options)
    
    def _get_rewards(self):
        return super()._get_rewards()
    
    def _get_observations(self):
        return super()._get_observations()