import numba as nb
from numba import cuda
import numpy as np
import math
from typing import Union
from types import ModuleType

try:
    from gpu_numba_solver.system_definition_template import *
except:
    from system_definition_template import *


# ---------------------------------------------------------------
#-----------------------  MAIN INTERFACE ------------------------
# ---------------------------------------------------------------

CUDA_JIT_PROPERTIES = {
    "fastmath"      : True,
    "opt"           : True,
    "max_registers" : 192,
    "debug"         : False,
    "lineinfo"      : False
}


# Try not to modify
__SYSTEM_DEF_FUNCTIONS = ["per_thread_ode_function",
                          "per_thread_action_after_timesteps",
                          "per_thread_initialization",
                          "per_thread_finalization",
                          "per_thread_event_function",
                          "per_thread_action_after_event_detection"]

def setup(system_definition: ModuleType, **kwargs):

    if hasattr(system_definition, "setup"):
        system_definition.setup(**kwargs)

    for func_name in __SYSTEM_DEF_FUNCTIONS:
        if hasattr(system_definition, func_name):
            globals()[func_name] = getattr(system_definition, func_name)



class SolverObject():

    def __init__(self, 
                number_of_threads : int,
                system_dimension  : int,
                number_of_control_parameters : int = 0,
                number_of_dynamic_parameters : int = 0,
                number_of_shared_parameters: int = 0,
                number_of_accessories : int = 1,
                number_of_events: int = 0,
                number_of_dense_outputs: int = 0,
                threads_per_block : int = 64,
                device_id: int = 0,
                method : str = "RKCK45",
                abs_tol : Union[list, float, None] = None,
                rel_tol : Union[list, float, None] = None,
                event_tol: Union[list, float, None] = None,
                event_dir: Union[list, int, None] = None,
                min_step: float = 1e-16,
                max_step: float = 1.0e6,
                init_step:float = 1e-2,
                growth_limit:float = 5.0,
                shrink_limit:float = 0.1,
                cud_jit_kwargs: Union[dict] = {}
                ):
        
        # ---- Device Properties and Setup ----------------
        num_devices = len(cuda.list_devices())
        assert num_devices > 0, "Error: GPU Device has not been found"
        assert device_id < num_devices, f"Error: GPU device with ID {device_id} is not available." 
        self._active_device = cuda.select_device(device_id)
        self._print_device_attributes()

        # ----- Constant (private) properties -------------
        self._system_dimension = system_dimension
        self._number_of_threads = number_of_threads
        self._number_of_control_parameters = number_of_control_parameters
        self._number_of_dynamic_parameters = number_of_dynamic_parameters
        self._number_of_shared_parameters = number_of_shared_parameters
        self._number_of_accessories = number_of_accessories
        self._number_of_events = number_of_events
        self._number_of_dense_outputs = number_of_dense_outputs
        self._threads_per_block = threads_per_block
        self._blocks_per_grid = self._number_of_threads // self._threads_per_block + (0 if self._number_of_threads % self._threads_per_block == 0 else 1 )
        self._method = method

        # ----- Constant (private) array sizes -----
        self._size_of_time_domain  = number_of_threads * 2
        self._size_of_actual_time  = number_of_threads
        self._size_of_actual_state = system_dimension * number_of_threads
        self._size_of_accessories  = max(number_of_accessories * number_of_threads, 1)
        self._size_of_control_parameters = max(number_of_control_parameters * number_of_threads, 1)
        self._size_of_dynamic_parameteters = max(number_of_dynamic_parameters * number_of_threads, 1)
        self._size_of_shared_parameters = max(self._number_of_shared_parameters, 1)
        self._size_of_dense_output_index = max(number_of_threads * number_of_dense_outputs, 1)
        self._size_of_dense_output_time_instances = max(number_of_threads * number_of_dense_outputs, 1)
        self._size_of_dense_output_states = max(number_of_threads * system_dimension * number_of_dense_outputs, 1)
        # TODO: Add more if necessary

        self._check_memory_usage()

        print("-------------------------------")
        print(f"Total number of Active Threads:    {self._number_of_threads:.0f}")
        print(f"BlockSize (threads per block):     {self._threads_per_block:.0f}")
        print(f"GridSize (total number of blocks): {self._blocks_per_grid:.0f}")


        # Tolerace workaround
        if abs_tol is None:
            self._abs_tol = np.full((system_dimension, ), 1e-8, dtype=np.float64)
        elif type(abs_tol) == list:
            self._abs_tol = np.array(abs_tol, dtype=np.float64)
        else:
            self._abs_tol = np.full((system_dimension, ), abs_tol, dtype=np.float64)

        if rel_tol is None:
            self._rel_tol = np.full((system_dimension, ), 1e-8, dtype=np.float64)
        elif type(rel_tol) == list:
            self._rel_tol = np.array(rel_tol, dtype=np.float64)
        else:
            self._rel_tol = np.full((system_dimension, ), rel_tol, dtype=np.float64)

        if event_tol is None:
            self._event_tol = np.full((max(number_of_events, 1), ), 1e-8, dtype=np.float64)
        elif type(event_tol) == list:
            self._event_tol = np.array(event_tol, dtype=np.float64)
        else:
            self._event_tol = np.full((max(number_of_events, 1), ), event_tol, dtype=np.float64)

        if event_dir is None:
            self._event_dir = np.full((max(number_of_events, 1), ), 0, dtype=np.int32)
        elif type(event_dir) == list:
            self._event_dir = np.array(event_dir, dtype=np.int32)
        else:
            self._event_dir = np.full((max(number_of_events, 1), ), event_dir, dtype=np.int32)


        assert len(self._abs_tol) == system_dimension
        assert len(self._rel_tol) == system_dimension
        assert len(self._event_tol) == max(number_of_events, 1)

        # ----- Public properties with default values ----
        self.time_step_init         = init_step
        self.time_step_max          = max_step
        self.time_step_min          = min_step
        self.time_step_growth_limit = growth_limit
        self.time_step_shrink_limit = shrink_limit

        # ---- Host side (private) arrays -----
        self._h_time_domain  = cuda.pinned_array((self._size_of_time_domain, ), dtype=np.float64)
        self._h_actual_time  = cuda.pinned_array((self._size_of_actual_time, ), dtype=np.float64)
        self._h_actual_state = cuda.pinned_array((self._size_of_actual_state, ), dtype=np.float64)
        self._h_accessories  = cuda.pinned_array((self._size_of_accessories, ), dtype=np.float64)
        self._h_control_parameters = cuda.pinned_array((self._size_of_control_parameters, ), dtype=np.float64)
        self._h_dynamic_parameters = cuda.pinned_array((self._size_of_dynamic_parameteters, ), dtype=np.float64)
        self._h_shared_parameters = cuda.pinned_array((self._size_of_shared_parameters, ), dtype=np.float64)
        self._h_status = cuda.pinned_array((number_of_threads, ), dtype=np.int8)
        self._h_dense_output_index = cuda.pinned_array((self._number_of_threads, ), dtype=np.int32)
        self._h_dense_output_time_instances = cuda.pinned_array((self._size_of_dense_output_time_instances, ), dtype=np.float64)
        self._h_dense_output_states = cuda.pinned_array((self._size_of_dense_output_states, ), dtype=np.float64)

        # ---- Device side (private) arrays -----
        self._d_time_domain  = cuda.device_array_like(self._h_time_domain)
        self._d_actual_time  = cuda.device_array_like(self._h_actual_time)
        self._d_actual_state = cuda.device_array_like(self._h_actual_state)
        self._d_accessories  = cuda.device_array_like(self._h_accessories)
        self._d_control_parameters = cuda.device_array_like(self._h_control_parameters)
        self._d_dynamic_parameters = cuda.device_array_like(self._h_dynamic_parameters)
        self._d_shared_parameters = cuda.device_array_like(self._h_shared_parameters)
        self._d_status = cuda.device_array_like(self._h_status)
        self._d_dense_output_index = cuda.device_array_like(self._h_dense_output_index)
        self._d_dense_output_time_instances = cuda.device_array_like(self._h_dense_output_time_instances)
        self._d_dense_output_states = cuda.device_array_like(self._h_dense_output_states)

        # ---- Create / Compile the Kenel function -----
        self._cuda_jit_kwargs = CUDA_JIT_PROPERTIES.copy()
        self._cuda_jit_kwargs.update(**cud_jit_kwargs)
        self._njit_cuda_kernel = _RKCK45_kernel(self._number_of_threads,
                                                self._system_dimension,
                                                self._number_of_control_parameters,
                                                self._number_of_dynamic_parameters,
                                                self._number_of_shared_parameters,
                                                self._number_of_accessories,
                                                self._number_of_events,
                                                self._number_of_dense_outputs,
                                                self._method,
                                                self._abs_tol,
                                                self._rel_tol,
                                                self._event_tol,
                                                self._event_dir,
                                                self._cuda_jit_kwargs)
        

    # ---- Interface to call the GPU Kernel function ------
    def solve_my_ivp(self):
        self._njit_cuda_kernel[self._blocks_per_grid, self._threads_per_block](
                                self._number_of_threads,
                                self._d_actual_state,
                                self._d_control_parameters,
                                self._d_dynamic_parameters,
                                self._d_shared_parameters,
                                self._d_accessories,
                                self._d_time_domain,
                                self._d_actual_time,
                                self._d_status,
                                self._d_dense_output_index,
                                self._d_dense_output_time_instances,
                                self._d_dense_output_states,
                                self.time_step_init,
                                self.time_step_max,
                                self.time_step_min,
                                self.time_step_growth_limit,
                                self.time_step_shrink_limit)
        cuda.synchronize()

    def status(self):
        """
        Algorithm termination:
            -1: integration failure,
             0: time_domain end is reached
            +1: termination event occured
        """
        self._d_status.copy_to_host(self._h_status)
        cuda.synchronize()

        return np.array(self._h_status, dtype=np.int8)


    def _print_device_attributes(self):

        print("-----------------------------------------")
        if not self._active_device:
            raise RuntimeError("No active CUDA device selected.")
        print(f"Device ID: {self._active_device.id}")
        print(f"Name: {self._active_device.name}")
        print(f"Compute Capability: {self._active_device.compute_capability[0]}.{self._active_device.compute_capability[1]}")
        print(f"Max Threads per Block: {self._active_device.MAX_THREADS_PER_BLOCK}")
        print(f"Max Block Dimensions: {self._active_device.MAX_BLOCK_DIM_X}, {self._active_device.MAX_BLOCK_DIM_Y}, {self._active_device.MAX_BLOCK_DIM_Z}")
        print(f"Max Grid Dimensions: {self._active_device.MAX_GRID_DIM_X}, {self._active_device.MAX_GRID_DIM_Y}, {self._active_device.MAX_GRID_DIM_Z}")
        print(f"Shared Memory per Block: {self._active_device.MAX_SHARED_MEMORY_PER_BLOCK // 1024} KB")
        print(f"Total Constant Memory: {self._active_device.TOTAL_CONSTANT_MEMORY // 1024} KB")
        print(f"Number of Multiprocessors: {self._active_device.MULTIPROCESSOR_COUNT}")
        print(f"Warp Size: {self._active_device.WARP_SIZE}")
        
        # Memory Info
        free_mem, total_mem = cuda.current_context().get_memory_info()
        print(f"Total Memory: {total_mem / (1024 ** 2):.2f} MB")
        print(f"Free Memory: {free_mem / (1024 ** 2):.2f} MB")


    def calculate_memory_requirements(self):
        bytes_per_element = 8
        total_elements = (
            self._size_of_time_domain +
            self._size_of_actual_time +
            self._size_of_actual_state +
            self._size_of_accessories +
            self._size_of_control_parameters +
            self._size_of_dynamic_parameteters +
            self._size_of_shared_parameters +
            self._size_of_dense_output_index +
            self._size_of_dense_output_time_instances +
            self._size_of_dense_output_states
        )
        return (total_elements * bytes_per_element, self._size_of_shared_parameters * bytes_per_element) 

    def _check_memory_usage(self):
        required_memory, required_shared_memory = self.calculate_memory_requirements()
        max_shared_memory = self._active_device.MAX_SHARED_MEMORY_PER_BLOCK
        try:
            free_memory, _ = cuda.current_context().get_memory_info()
        except cuda.CudaSupportError:
            raise RuntimeError("No active CUDA context found. Please ensure a valid GPU is selected.")

        required_memory_MB = required_memory / (1024 ** 2)
        free_memory_MB = free_memory / (1024 ** 2)

    
        print(f"Required memory: {required_memory_MB:.2f} MB")
        print(f"Free GPU memory: {free_memory_MB:.2f} MB")
        print("-----------------------------------------")
        if required_memory > free_memory:
            raise ValueError(
                f"Error: The required memory usge ({required_memory_MB:.2f} MB) exceeds "
                f"the device's available memory ({free_memory_MB:.2f} MB)."
            )
        else:
            print(f"Global Memory check passed: The required memory fits within the available GPU memory: {required_memory_MB:.2f}/{free_memory_MB:.2f} MB")


        if required_shared_memory > max_shared_memory:
            raise ValueError(
                f"Error: The shared memory usage ({required_shared_memory / 1024:.2f} KB) exceeds "
                f"the device's maximum shared memory ({max_shared_memory / 1024:.2f} KB)."
            )
        else:
            print(f"Shared Memory check passed: The Shared memory usage is within limits: {required_shared_memory / 1024:.2f}/{max_shared_memory / 1024:.2f} KB")


    def syncronize(self):
        cuda.synchronize()

    # -------- Properties --------
    # ----------------------------
    def get_device_array(self, property: str, index: int):

        if property == "shared_parameters":
            return self._d_shared_parameters
        else:
            idx0 = index * self._number_of_threads
            idxN = idx0  + self._number_of_threads
            if property == "time_domain":
                pass
            elif property == "actual_state":
                return self._d_actual_state[idx0:idxN]
            elif property == "control_parameters":
                return self._d_control_parameters[idx0:idxN]
            elif property == "dynamic_parameters":
                return self._d_dynamic_parameters[idx0:idxN]
            elif property == "accessories":
                return self._d_accessories[idx0:idxN]
        
              
    def set_device_array(self, property: str, index: int, d_ary):

        if property == "shared_parameters":
            pass
        else:
            idx0 = index * self._number_of_threads
            idxN = idx0  + self._number_of_threads
            assert len(d_ary) == self._number_of_threads, print("Err: The length of the array does not match with the number of thread")

            if property == "time_domain":
                pass
            elif property == "actual_state":
                self._d_actual_state[idx0:idxN].copy_to_device(cuda.as_cuda_array(d_ary))
            elif property == "control_parameters":
                pass
            elif property == "dynamic_parameters":
                self._d_dynamic_parameters[idx0:idxN].copy_to_device(cuda.as_cuda_array(d_ary))
            elif property == "accessories":
                pass


    # ----  Helper functions -----
    #-----------------------------
    def set_device(self, thread_id: int, property: str, index, value: float):

        idx = thread_id + index * self._number_of_threads

        if property == "time_domain":
            self._d_time_domain[idx] = np.float64(value)
        elif property == "actual_state":
            self._d_actual_state[idx] = np.float64(value)
        elif property == "control_parameters":
            self._d_control_parameters[idx] = np.float64(value)
        elif property == "dynamic_parameters":
            self._d_dynamic_parameters[idx] = np.float64(value)
        elif property == "accessories":
            self._d_accessories[idx] = np.float64(value)
        else:
            print("Error: set_device")
     
    def set_host(self, thread_id: int, property: str, index: int, value: float) :

        idx = thread_id + index * self._number_of_threads
        
        if property == "time_domain":
            self._h_time_domain[idx] = np.float64(value)
        elif property == "actual_state":
            self._h_actual_state[idx] = np.float64(value)
        elif property == "control_parameters":
            self._h_control_parameters[idx] = np.float64(value)
        elif property == "dynamic_parameters":
            self._h_dynamic_parameters[idx] = np.float64(value)
        elif property == "accessories":
            self._h_accessories[idx] = np.float64(value)
        else:
            print("Error: SetHost")

    def set_shared_host(self, property: str, index, value):
        if property == "shared_parameters":
            self._h_shared_parameters[index] = np.float64(value)
        else:
            print("Error: Sethost. Not a valid property.")

    def get_host(self, thread_id: int, property: str, index: int):

        idx = thread_id + index * self._number_of_threads

        if property == "time_domain":
            return self._h_time_domain[idx]
        elif property == "actual_state":
            return self._h_actual_state[idx]
        elif property == "control_parameters":
            return self._h_control_parameters[idx]
        elif property == "dynamic_parameters":
            return self._h_control_parameters[idx]
        elif property == "accessories":
            return self._h_accessories[idx]
        else:
            print("Error: GetHost")
            return None

    def get_host_array(self, property: str, index: int):
        if property == "shared_parameters":
            return self._h_shared_parameters
        else:
            idx0 = index * self._number_of_threads
            idxN = idx0  + self._number_of_threads
            if property == "time_domain":
                pass
            elif property == "actual_state":
                return self._h_actual_state[idx0:idxN]
            elif property == "control_parameters":
                return self._h_control_parameters[idx0:idxN]
            elif property == "dynamic_parameters":
                return self._h_dynamic_parameters[idx0:idxN]
            elif property == "accessories":
                return self._h_accessories[idx0:idxN]

    def get_shared_host(self, property: str, index):
        if property == "shared_parameters":
            return self._h_shared_parameters[index]
        else:
            print("Error: Gethost. Not a valid property.")

    def get_dense_output(self):
        dense_index = self._h_dense_output_index.reshape((self._number_of_threads, ))
        dense_time = self._h_dense_output_time_instances.reshape((self._number_of_dense_outputs, self._number_of_threads))
        dense_states = self._h_dense_output_states.reshape((self._number_of_dense_outputs, self._system_dimension, self._number_of_threads))
        return dense_index, dense_time, dense_states

    def syncronize_h2d(self, property:str):
        if property == "time_domain":
            self._d_time_domain.copy_to_device(self._h_time_domain)
        elif property == "control_parameters":
            self._d_control_parameters.copy_to_device(self._h_control_parameters)
        elif property == "dynamic_parameters":
            self._d_dynamic_parameters.copy_to_device(self._h_dynamic_parameters)
        elif property == "shared_parameters":
            self._d_shared_parameters.copy_to_device(self._h_shared_parameters)
        elif property == "actual_state":
            self._d_actual_state.copy_to_device(self._h_actual_state)
        elif property == "accessories":
            self._d_accessories.copy_to_device(self._h_accessories)
        elif property == "dense_output":
            self._d_dense_output_index.copy_to_device(self._h_dense_output_index)
            self._d_dense_output_time_instances.copy_to_device(self._h_dense_output_time_instances)
            self._d_dense_output_states.copy_to_device(self._h_dense_output_states)
        elif property == "all":
            self._d_time_domain.copy_to_device(self._h_time_domain)
            self._d_control_parameters.copy_to_device(self._h_control_parameters)
            self._d_dynamic_parameters.copy_to_device(self._h_dynamic_parameters)
            self._d_shared_parameters.copy_to_device(self._h_shared_parameters)
            self._d_actual_state.copy_to_device(self._h_actual_state)
            self._d_accessories.copy_to_device(self._h_accessories)
            self._d_dense_output_index.copy_to_device(self._h_dense_output_index)
            self._d_dense_output_time_instances.copy_to_device(self._h_dense_output_time_instances)
            self._d_dense_output_states.copy_to_device(self._h_dense_output_states)
        else:
            print("Error...")

        cuda.synchronize()

    def syncronize_d2h(self, property:str):
        if property == "time_domain":
            self._d_time_domain.copy_to_host(self._h_time_domain)
        elif property == "control_parameters":
            self._d_control_parameters.copy_to_host(self._h_control_parameters)
        elif property == "dynamic_parameters":
            self._d_dynamic_parameters.copy_to_host(self._h_dynamic_parameters)
        elif property == "shared_parameters":
            self._d_shared_parameters.copy_to_host(self._h_shared_parameters)
        elif property == "actual_state":
            self._d_actual_state.copy_to_host(self._h_actual_state)
        elif property == "accessories":
            self._d_accessories.copy_to_host(self._h_accessories)
        elif property == "dense_output":
            self._d_dense_output_index.copy_to_host(self._h_dense_output_index)
            self._d_dense_output_time_instances.copy_to_host(self._h_dense_output_time_instances)
            self._d_dense_output_states.copy_to_host(self._h_dense_output_states)
        elif property == "all":
            self._d_time_domain.copy_to_host(self._h_time_domain)
            self._d_control_parameters.copy_to_host(self._h_control_parameters)
            self._d_dynamic_parameters.copy_to_host(self._h_dynamic_parameters)
            self._d_shared_parameters.copy_to_host(self._h_shared_parameters)
            self._d_actual_state.copy_to_host(self._h_actual_state)
            self._d_accessories.copy_to_host(self._h_accessories)
            self._d_dense_output_index.copy_to_host(self._h_dense_output_index)
            self._d_dense_output_time_instances.copy_to_host(self._h_dense_output_time_instances)
            self._d_dense_output_states.copy_to_host(self._h_dense_output_states)
        else:
            print("Error...")

        cuda.synchronize()


# ----------  ALGO ---------------
# --------------------------------
# ----- Main Kernel Function -----
def _RKCK45_kernel( number_of_threads: int,
                    system_dimension: int,
                    number_of_control_parameters: int,
                    number_of_dynamic_parameters: int,
                    number_of_shared_parameters: int,
                    number_of_accessories: int,
                    number_of_events: int,
                    number_of_dense_outputs: int,
                    method : str,
                    abs_tol : np.ndarray,
                    rel_tol : np.ndarray,
                    event_tol : np.ndarray,
                    event_dir : np.ndarray,
                    cuda_jit: dict):
    
    # CONSTANT PARAMETERS ---------------
    # Create alieses
    NT  = number_of_threads
    SD  = system_dimension
    NCP = number_of_control_parameters
    NDP = number_of_dynamic_parameters
    NSP = number_of_shared_parameters
    NA  = number_of_accessories
    NE  = number_of_events
    NDO = number_of_dense_outputs
    ALGO = method 
    ATOL = abs_tol
    RTOL = rel_tol
    ETOL = event_tol
    EDIR = event_dir
    CUDA_JIT = cuda_jit

    # DEVICE FUNCTIONS CALLED IN THE KERNEL
    # -------------------------------------

    # Error Control -----------------------
    @cuda.jit(nb.types.Tuple(
            [nb.boolean, nb.boolean, nb.float64])(
                nb.int32,
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                nb.boolean,
                nb.boolean
    ), device=True, inline=True)
    def per_thread_error_control_RKCK45(tid,
                                 l_next_state,
                                 l_actual_state,
                                 l_error,
                                 l_time_step,
                                 time_step_max,
                                 time_step_min,
                                 time_step_growth_limit,
                                 time_step_shrink_limit,
                                 l_update_step,
                                 l_is_finite):
        
        relative_error = 1e30
        l_terminate_thread = False
        for i in range(SD):
            error_tolerance =  max( RTOL[i]*max( abs(l_next_state[i]), abs(l_actual_state[i])), ATOL[i] )
            l_update_step   = l_update_step and ( l_error[i] < error_tolerance )
            relative_error  = min(relative_error, error_tolerance / l_error[i])

        if l_update_step:
            time_step_multiplicator = 0.9 * math.pow(relative_error, (1.0/5.0))
        else:
            time_step_multiplicator = 0.9 * math.pow(relative_error, (1.0/4.0))

        if math.isfinite(time_step_multiplicator) == False:
            print('Thread id ',tid,'State is not finite')
            l_is_finite = False


        if l_is_finite == False:
            time_step_multiplicator = time_step_shrink_limit
            l_update_step = False

            if l_time_step < time_step_min*1.01:
                l_terminate_thread = True
                print('Thread id ',tid,'State is not a finite number even with the minimal step size')

        else:
            if (l_time_step < time_step_min*1.01):
                print('Thread id ',tid,': Minimum time step is reached')
                l_update_step = False
                l_terminate_thread = True

        time_step_multiplicator = min(time_step_multiplicator, time_step_growth_limit)
        time_step_multiplicator = max(time_step_multiplicator, time_step_shrink_limit)

        l_new_time_step = l_time_step * time_step_multiplicator
        l_new_time_step = min(l_new_time_step, time_step_max)
        l_new_time_step = max(l_new_time_step, time_step_min)

        return l_update_step, l_terminate_thread, l_new_time_step

    # Event Handling -----------------------
    @cuda.jit(nb.types.Tuple(
        [nb.boolean, nb.float64])(
            nb.int32,
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.boolean,
            nb.boolean
    ), device=True, inline=True)
    def per_thread_event_time_step_control(tid,
                                           l_actual_event_value,
                                           l_next_event_value,
                                           l_time_step,
                                           l_new_time_step,
                                           l_time_step_min,
                                           l_update_step,
                                           l_terminate_thread):
        
        l_event_time_step = l_time_step
        l_is_corrected = False

        if( (l_update_step == True) and 
            (l_terminate_thread == False)):
            for i in range(NE):
                if( ( ( l_actual_event_value[i] >  ETOL[i] ) and ( l_next_event_value[i] < -ETOL[i] ) and ( EDIR[i] <= 0 ) ) or
				    ( ( l_actual_event_value[i] < -ETOL[i] ) and ( l_next_event_value[i] >  ETOL[i] ) and ( EDIR[i] >= 0 ) ) ):
                        l_event_time_step = min( l_event_time_step, -l_actual_event_value[i] / (l_next_event_value[i]-l_actual_event_value[i]) * l_time_step )
                        l_is_corrected = True
                        

        if l_is_corrected == True:
            if l_event_time_step < l_time_step_min:
                print("Warning (tid: ", tid, ") : Event can not be detected without reducing the step size below the")
            else:
                l_new_time_step = l_event_time_step
                l_update_step = False

        return l_update_step, l_new_time_step

    # Dense Output --------------------------
    @cuda.jit(nb.types.Tuple(
            [nb.float64, nb.int32])(
            nb.int32,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.int32
    ), device=True, inline=True)
    def per_thread_store_dense_output(tid,
                                      dense_output_time_instances,
                                      dense_output_states,
                                      l_actual_state,
                                      l_actual_time,
                                      l_dense_output_time,
                                      l_dense_output_min_time_step,
                                      l_time_domain_end,
                                      l_dense_output_index,
                                      ):
        
        dense_output_time_instances[tid + l_dense_output_index * NT] = l_actual_time
        dense_output_state_index = tid + l_dense_output_index * NT * SD
        for i in nb.prange(SD):
            dense_output_states[dense_output_state_index] = l_actual_state[i]
            dense_output_state_index += NT

        l_dense_output_index += 1
        l_dense_output_time = min(l_actual_time + l_dense_output_min_time_step, l_time_domain_end)

        return l_dense_output_time, l_dense_output_index 

    # STEPPER FUNCTIONS -----------------------------
    @cuda.jit(nb.boolean(
                nb.int32,
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64,
                nb.float64
    ), device=True, inline=True)
    def per_thread_stepper_RK45(tid,
                            l_next_state,
                            l_actual_state,
                            l_error,
                            l_control_parameter,
                            l_dynamic_parameter,
                            s_shared_parameter,
                            l_accessories,
                            l_actual_time,
                            l_time_step):
        
        l_isfinite = True
        k1 = cuda.local.array((SD, ), dtype=nb.float64)
        x  = cuda.local.array((SD, ), dtype=nb.float64)
        dTp2 = 0.5 * l_time_step
        dTp6 = 1.0 / 6.0 * l_time_step

        # K1 ---------------------------
        per_thread_ode_function(tid,
                                l_actual_time,
                                l_next_state,
                                l_actual_state,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # K2 ----------------------------
        t = l_actual_time + dTp2

        for i in nb.prange(SD):
            x[i] = l_actual_state[i] + l_next_state[i] * dTp2

        per_thread_ode_function(tid,
                                t,
                                k1,
                                x, 
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # K3 ----------------------------
        for i in nb.prange(SD):
            l_next_state[i] = l_next_state[i] + 2.0 * k1[i]
            x[i] = l_actual_state[i] + k1[i] * dTp2

        per_thread_ode_function(tid,
                                t,
                                k1,
                                x,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)

        # K4 -------------------------------
        t = l_actual_time + l_time_step

        for i in nb.prange(SD):
            l_next_state[i] = l_next_state[i] + 2.0 * k1[i]
            x[i] = l_actual_state[i] + k1[i] * l_time_step

        per_thread_ode_function(tid,
                                t,
                                k1,
                                x,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)

        # NEW STATE -------------------------
        for i in nb.prange(SD):
            l_next_state[i] = l_actual_state[i] + dTp6 * (l_next_state[i] + k1[i])

            if math.isfinite(l_next_state[i]) == False:
                l_isfinite = False

        return l_isfinite

    @cuda.jit(nb.boolean(
                nb.int32,
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64,
                nb.float64
    ), device=True, inline=True)
    def per_thread_stepper_RKCK45(tid,
                           l_next_state,
                           l_actual_state,
                           l_error,
                           l_control_parameter,
                           l_dynamic_parameter,
                           s_shared_parameter,
                           l_accessories,
                           l_actual_time,
                           l_time_step):
        
        k1 = cuda.local.array((SD, ), dtype=nb.float64)
        k2 = cuda.local.array((SD, ), dtype=nb.float64)
        k3 = cuda.local.array((SD, ), dtype=nb.float64)
        k4 = cuda.local.array((SD, ), dtype=nb.float64)
        k5 = cuda.local.array((SD, ), dtype=nb.float64)
        k6 = cuda.local.array((SD, ), dtype=nb.float64)
        x  = cuda.local.array((SD, ), dtype=nb.float64)
        t  = nb.float64(l_actual_time)
        l_is_finite = True

        # K1 --------------------
        per_thread_ode_function(tid,
                                l_actual_time,
                                k1,
                                l_actual_state,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)


        # K2 ---------------------
        t = l_actual_time + l_time_step * (1.0 / 5.0)
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] + l_time_step * (1.0 / 5.0) * k1[i]
        
        per_thread_ode_function(tid,
                                t,
                                k2,
                                x, 
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # K3 --------------------
        t = l_actual_time + l_time_step * (3.0 / 10.0)
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] \
                    + l_time_step * ( (3.0 / 40.0) * k1[i] \
                                  +   (9.0 / 40.0) * k2[i] )

        per_thread_ode_function(tid,
                                t,
                                k3,
                                x,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # K4 -------------------
        t = l_actual_time + l_time_step * (3.0 / 5.0)
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] \
                    + l_time_step * ( (3.0 / 10.0) * k1[i] \
                                    - (9.0 / 10.0) * k2[i] \
                                    + (6.0 / 5.0) * k3[i] )
            
        per_thread_ode_function(tid,
                                t,
                                k4,
                                x, 
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # K5 -------------------
        t = l_actual_time + l_time_step
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] \
                    + l_time_step * (-(11.0 / 54.0) * k1[i] \
                                    + (5.0 / 2.0 )  * k2[i] \
                                    - (70.0 / 27.0) * k3[i] \
                                    + (35.0 / 27.0) * k4[i] )

        per_thread_ode_function(tid,
                                t,
                                k5,
                                x,
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)

        # K6 ---------------------
        t = l_actual_time + l_time_step * (7.0 / 8.0)
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] \
                    + l_time_step * ((1631.0/55296.0) * k1[i] \
                                    + (175.0/512.0)   * k2[i] \
                                    + (575.0/13824.0) * k3[i] \
                                    + (44275.0/110592.0) * k4[i] \
                                    + (253.0/4096.0)  * k5[i] )

        per_thread_ode_function(tid,
                                t,
                                k6,
                                x, 
                                l_accessories,
                                l_control_parameter,
                                l_dynamic_parameter,
                                s_shared_parameter)
        
        # NEW STATE AND ERROR -------------------------
        for i in nb.prange(SD):
            l_next_state[i] = l_actual_state[i] \
                            + l_time_step * ( (37.0/378.0)  * k1[i] \
                                            + (250.0/621.0) * k3[i] \
                                            + (125.0/594.0) * k4[i] \
                                            + (512.0/1771.0) * k6[i] )

            l_error[i] = (  37.0/378.0  -  2825.0/27648.0 ) * k1[i] \
		               + ( 250.0/621.0  - 18575.0/48384.0 ) * k3[i] \
					   + ( 125.0/594.0  - 13525.0/55296.0 ) * k4[i] \
					   + (   0.0        -   277.0/14336.0 ) * k5[i] \
					   + ( 512.0/1771.0 -     1.0/4.0     ) * k6[i]

            l_error[i] = l_time_step * abs( l_error[i] ) + 1e-18
            
            if (math.isfinite(l_next_state[i]) == False or math.isfinite(l_error[i]) == False ):
                l_is_finite = False
        return l_is_finite

    # -------------------------------------------------
    # KERNEL FUNCTION ---------------------------------
    # -------------------------------------------------
    @cuda.jit(nb.void(
                nb.int32,  
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64[:],
                nb.int8[:],
                nb.int32[:],
                nb.float64[:],
                nb.float64[:],
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                ),
                **CUDA_JIT)
    def single_system_per_thread(number_of_threads,
                                actual_state,
                                control_parameter,
                                dynamic_parameter,
                                shared_parameter,
                                accessories,
                                time_domain,
                                actual_time,
                                status,
                                dense_output_index,
                                dense_output_time_instances,
                                dense_output_states,
                                time_step_init,
                                time_step_max,
                                time_step_min,
                                time_step_growth_limit,
                                time_step_shrink_limit):
        
        tid = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x

        # SHARED MEMORY MANAGEMENT
        s_shared_parameter = cuda.shared.array((NSP, ) if NSP !=0 else (1, ), dtype=nb.float64)
        num_sp_fill = NSP // cuda.blockDim.x + (0 if NSP % cuda.blockDim.x == 0 else 1)
        for i in nb.prange(num_sp_fill):
            s_tid = cuda.threadIdx.x + i * cuda.blockDim.x
            if s_tid < NSP:
                s_shared_parameter[s_tid] = shared_parameter[s_tid]

        cuda.syncthreads()

        # RKCK 45 algo
        if tid < number_of_threads:

            # THREAD LOCAL MEMORY
            l_time_domain = cuda.local.array((2, ), dtype=nb.float64)
            l_actual_state = cuda.local.array((SD, ), dtype=nb.float64)
            l_next_state = cuda.local.array((SD, ), dtype=nb.float64)
            l_error = cuda.local.array((SD, ), dtype=nb.float64)
            l_control_parameter = cuda.local.array((NCP, ) if NCP !=0 else (1, ), dtype=nb.float64)
            l_dynamic_parameter = cuda.local.array((NDP, ) if NDP !=0 else (1, ), dtype=nb.float64)
            l_accessories = cuda.local.array((NA, ) if NA !=0 else (1, ), dtype=nb.float64)
            l_actual_event_value = cuda.local.array((NE, ) if NE !=0 else (1, ), dtype=nb.float64)
            l_next_event_value = cuda.local.array((NE, ) if NE !=0 else (1, ), dtype=nb.float64)
            l_status = 0

            for i in nb.prange(2):
                l_time_domain[i] = time_domain[tid + i*NT]
            
            for i in nb.prange(SD):
                l_actual_state[i] = actual_state[tid + i*NT]

            for i in nb.prange(NCP):
                l_control_parameter[i] = control_parameter[tid + i*NT]

            for i in nb.prange(NDP):
                l_dynamic_parameter[i] = dynamic_parameter[tid + i*NT]

            for i in nb.prange(NA):
                l_accessories[i] = accessories[tid + i*NT]
            
            l_actual_time = l_time_domain[0]
            l_time_step = time_step_init
            l_new_time_step = time_step_init
            l_terminate = False    
            l_event_terminal = False 

            if NDO > 0:
                l_dense_output_time = l_time_domain[0]
                l_dense_output_min_time_step = (l_time_domain[1] - l_time_domain[0]) / (NDO - 1)
                l_dense_output_time, \
                l_dense_output_index = per_thread_store_dense_output(
                                        tid,
                                        dense_output_time_instances,
                                        dense_output_states,
                                        l_actual_state,
                                        l_actual_time,
                                        l_dense_output_time,
                                        l_dense_output_min_time_step,
                                        l_time_domain[1],
                                        0)          
    
            # INITIALIZATION
            per_thread_initialization(
                            tid,
                            l_actual_time,
                            l_time_domain,
                            l_actual_state,
                            l_accessories,
                            l_control_parameter,
                            l_dynamic_parameter,
                            s_shared_parameter)
            
            if NE > 0:
                per_thread_event_function(  tid,
                                            l_actual_time,
                                            l_actual_event_value,
                                            l_actual_state,
                                            l_accessories,
                                            l_control_parameter,
                                            l_dynamic_parameter,
                                            s_shared_parameter)

            cuda.syncthreads()
            while l_terminate == False:
                
                # INITIALIZE TIME STEP --------------
                l_update_step = True
                l_time_domain_ends = False 

                l_time_step = l_new_time_step
                if (l_time_step > (l_time_domain[1] - l_actual_time)):
                    l_time_step = l_time_domain[1] - l_actual_time
                    l_time_domain_ends = True


                # FORWARD STEPPING ----------------
                if ALGO == "RK45":
                    l_is_finite = per_thread_stepper_RK45(
                                    tid,
                                    l_next_state,
                                    l_actual_state,
                                    l_error,
                                    l_control_parameter,
                                    l_dynamic_parameter,
                                    s_shared_parameter,
                                    l_accessories,
                                    l_actual_time,
                                    l_time_step)
                
                    if l_is_finite == False:
                        print("State is not finite")
                        l_terminate = 1
                        l_new_time_step = time_step_init
                        l_status = -1

                elif ALGO == "RKCK45":
                    l_is_finite = per_thread_stepper_RKCK45(
                                    tid,
                                    l_next_state,
                                    l_actual_state,
                                    l_error,
                                    l_control_parameter,
                                    l_dynamic_parameter,
                                    s_shared_parameter,
                                    l_accessories,
                                    l_actual_time,
                                    l_time_step)


                    l_update_step, \
                    l_terminate,   \
                    l_new_time_step = per_thread_error_control_RKCK45(
                                        tid,
                                        l_next_state,
                                        l_actual_state,
                                        l_error,
                                        l_time_step,
                                        time_step_max,
                                        time_step_min,
                                        time_step_growth_limit,
                                        time_step_shrink_limit,
                                        l_update_step,
                                        l_is_finite)
                    if l_terminate:
                        l_status = -1

                # NEW EVENT VALUE AND TIME STEP CONTROL
                if NE > 0:
                    per_thread_event_function(
                                        tid,
                                        l_actual_time+l_time_step,
                                        l_next_event_value,
                                        l_next_state,
                                        l_accessories,
                                        l_control_parameter,
                                        l_dynamic_parameter,
                                        s_shared_parameter)

                    l_update_step, \
                    l_new_time_step = per_thread_event_time_step_control(
                                        tid,
                                        l_actual_event_value,
                                        l_next_event_value,
                                        l_time_step,
                                        l_new_time_step,
                                        time_step_min,
                                        l_update_step,
                                        l_terminate)
                       
                # SUCCESFULL TIMESTEPPING UPDATE TIME AND STATE
                if l_update_step == True:
                    l_actual_time += l_time_step

                    for i in nb.prange(SD):
                        l_actual_state[i] = l_next_state[i]

                    # ACTION AFTER SUCCESFULL TIMESTEP
                    l_user_terminal = per_thread_action_after_timesteps(
                                        tid,
                                        l_actual_time,
                                        l_actual_state,
                                        l_accessories,
                                        l_control_parameter,
                                        l_dynamic_parameter,
                                        s_shared_parameter)

                    if NE > 0:
                        l_event_terminal = False
                        for i in range(NE):
                            if( ( ( l_actual_event_value[i] >  ETOL[i] ) and ( abs(l_next_event_value[i]) < ETOL[i] ) and ( EDIR[i] <= 0 ) ) or
							    ( ( l_actual_event_value[i] < -ETOL[i] ) and ( abs(l_next_event_value[i]) < ETOL[i] ) and ( EDIR[i] >= 0 ) ) ):
                                if per_thread_action_after_event_detection(
                                    tid,
                                    i,
                                    l_actual_time,
                                    l_time_domain,
                                    l_actual_state,
                                    l_accessories,
                                    l_control_parameter,
                                    l_dynamic_parameter,
                                    s_shared_parameter):
                                    l_event_terminal = True
                                    l_status = 1
                                
                        per_thread_event_function(
                            tid,
                            l_actual_time,
                            l_actual_event_value,
                            l_actual_state,
                            l_accessories,
                            l_control_parameter,
                            l_dynamic_parameter,
                            s_shared_parameter)

                    if NDO > 0:
                        # Check Storage Condition
                        if ((l_dense_output_index < NDO) and (l_dense_output_time <= l_actual_time)):
                            l_dense_output_time, \
                            l_dense_output_index = per_thread_store_dense_output(
                                                        tid,
                                                        dense_output_time_instances,
                                                        dense_output_states,
                                                        l_actual_state,
                                                        l_actual_time,
                                                        l_dense_output_time,
                                                        l_dense_output_min_time_step,
                                                        l_time_domain[1],
                                                        l_dense_output_index)                                

                    if l_time_domain_ends or l_event_terminal or l_user_terminal:
                        l_terminate = True

            # FINALIZATION ------------------------------------
            per_thread_finalization(
                            tid,
                            l_actual_time,
                            l_time_domain,
                            l_actual_state,
                            l_accessories,
                            l_control_parameter,
                            l_dynamic_parameter,
                            s_shared_parameter)

            cuda.syncthreads()
            # Integration ends, wrtite data back to global memory
            for i in nb.prange(2):
                time_domain[tid + i*NT] = l_time_domain[i]

            for i in nb.prange(SD):
                actual_state[tid + i*NT] = l_actual_state[i]

            for i in nb.prange(NCP):
                control_parameter[tid + i*NT] = l_control_parameter[i]

            for i in nb.prange(NDP):
                dynamic_parameter[tid +i*NT] = l_dynamic_parameter[i]

            for i in nb.prange(NA):
                accessories[tid + i*NT] = l_accessories[i]

            actual_time[tid] = l_actual_time
            status[tid] = l_status
            dense_output_index[tid] = l_dense_output_index if NDO > 0 else 0



    return single_system_per_thread


# ----------------------------------
