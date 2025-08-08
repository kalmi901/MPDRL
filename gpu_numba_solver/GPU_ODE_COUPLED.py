"""
ODE Solver for medium-sized coupled ode-systems

"""
import numba as nb
from numba import cuda
import numpy as np
import math
from types import ModuleType
from typing import Dict, Optional

try:
    from gpu_numba_solver._coupled_system_definition_template import *
    from gpu_numba_solver.system_definitions.system_registry import registered_systems, discover_systems_in_package
except:
    from _coupled_system_definition_template import *
    from system_definitions.system_registry import registered_systems, discover_systems_in_package

# ----------------------------------------------------------------
# ---------------------- MAIN INTERFACE --------------------------
# ----------------------------------------------------------------

CUDA_JIT_PROPERTIES = {
    "fastmath"      : True,
    "opt"           : True,
    "max_registers" : 128,
    "debug"         : False,
    "lineinfo"      : False
}


# Try not to modify
# TODO ----- >
__SYSTEM_DEF_FUNCTIONS = ["per_block_ode_function",
                          "per_block_initialization",
                          "per_block_finalization",
                          "per_thread_action_after_timesteps",
                          "per_thread_finalization",
                          "per_thread_event_function",
                          "per_thread_action_after_event_detection"]


discover_systems_in_package("./gpu_numba_solver/system_definitions", "system_definitions")

def setup(model_name: str, **kwargs):
    if model_name not in registered_systems:
        raise ValueError(f"Model {model_name} not found")
    
    model_cls = registered_systems[model_name]

    # Call the model setup first!!!
    # Some global functins are initialized during setup
    if hasattr(model_cls, "setup"):
        model_cls.setup(**kwargs)

    # Put device functions on the global scope
    for func_name in __SYSTEM_DEF_FUNCTIONS:
        if hasattr(model_cls, "functions") and func_name in model_cls.functions:
            print(func_name)
            globals()[func_name] = model_cls.functions[func_name]

    return model_cls.defaults, model_cls.parameters



def _init_array(val, shape, dtype, default):
    if val is None:
        return np.full(shape, default, dtype=dtype)
    elif isinstance(val, list):
        return np.array(val, dtype=dtype)
    else:
        return np.full(shape, val, dtype=dtype)

class CoupledSolverObject:
    def __init__(self,
                number_of_systems           : int,               # N  - the total number of solved coupled systems
                number_of_sytems_per_block  : int,               # Nb - the number of coupled systems per threadblock 
                number_of_units_per_system  : int,               # n  - number of ODE replicases 
                unit_system_dimension       : int,               # NS - single ODE system dimension
                number_of_unit_parameters   : int = 0,           # per unit Control parameter
                number_of_system_parameters : int = 0,           # per systes
                number_of_global_parameters : int = 0,           # common for all instances
                number_of_dynamic_parameters: int = 0,           # per systems
                number_of_unit_accessories  : int = 0,           # per units
                number_of_system_accessories: int = 0,           # per systems
                number_of_coupling_matrices: int = 1,            # NC coupling matrices per system
                number_of_coupling_terms: int = 1,               # NCT 
                number_of_coupling_factors: int = 1,             # NCF
                number_of_events: int = 0,
                number_of_dense_outputs: int = 0,
                threads_per_block : int = 128,
                device_id: int = 0,
                method : str = "RKCK45",
                linsolve: str = "BICGSTAB",
                abs_tol : list | float | None = None,
                rel_tol : list | float | None = None,
                event_tol: list | float| None = None,
                event_dir:  list | int | None = None,
                min_step: float = 1e-16,
                max_step: float = 1.0e6,
                init_step:float = 1e-2,
                growth_limit:float = 5.0,
                shrink_limit:float = 0.1,
                cuda_jit_kwargs: Optional[Dict] = None
                ):

        # ---- Device Properties and Setup ----------------
        num_devices = len(cuda.list_devices())
        assert num_devices > 0, "Error: GPU Device has not been found"
        assert device_id < num_devices, f"Error: GPU device with ID {device_id} is not available." 
        self._active_device = cuda.select_device(device_id)
        self._print_device_attributes()


        # ----- Constant (private) properties -----
        self._number_of_systems = number_of_systems
        self._number_of_systems_per_block = number_of_sytems_per_block
        self._number_of_units_per_system = number_of_units_per_system
        self._unit_system_dimension = unit_system_dimension
        self._number_of_unit_paramenters = number_of_unit_parameters
        self._number_of_system_parameters = number_of_system_parameters
        self._number_of_global_parameters = number_of_global_parameters
        self._number_dynamic_parameters = number_of_dynamic_parameters
        self._number_of_unit_accessories = number_of_unit_accessories
        self._number_of_system_accessories = number_of_system_accessories
        self._number_of_coupling_matrices = number_of_coupling_matrices
        self._number_of_coupling_terms = number_of_coupling_terms
        self._number_of_coupling_factors = number_of_coupling_factors
        self._number_of_events = number_of_events
        self._number_of_dense_outputs = number_of_dense_outputs

        assert method in ["RKCK45"], f"Error: {method} is not supported"
        assert linsolve in ["BICGSTAB"], f"Error: {linsolve} is not supported"
        self._method = method
        self._linsolve = linsolve
        
        # --- Thread Management ---
        logical_threads_per_block = self._number_of_systems_per_block * self._number_of_units_per_system  
        # Hány CUDA blokk kell ahhoz, hogy ezt a logikai blokkot teljesen feldolgozzuk?
        number_of_block_launches = logical_threads_per_block // threads_per_block + (
            0 if logical_threads_per_block % threads_per_block == 0 else 1 )
        # Padding: ha nem osztható TPB-vel, pótoljuk dummy szálakkal
        thread_padding = number_of_block_launches * threads_per_block - logical_threads_per_block
        # Hány CUDA blokk kell összesen, hogy az összes rendszer le legyen fedve?
        blocks_per_grid = number_of_systems // self._number_of_systems_per_block + (
            0 if self._number_of_systems % self._number_of_systems_per_block == 0 else 1 )
        # Teljes szálmennyiség, az összes CUDA blokk és padding figyelembevételével
        total_logical_threads = (logical_threads_per_block + thread_padding) * blocks_per_grid 
        
        self.thread_config = {
            "logical_threads_per_block" : logical_threads_per_block,
            "number_of_block_launches"  : number_of_block_launches,
            "thread_padding"            : thread_padding,
            "blocks_per_grid"           : blocks_per_grid,
            "total_logical_threads"     : total_logical_threads,
            "threads_per_block"         : threads_per_block
        }


        print("-------------------------------")
        print(f"Total number of Systems:           {number_of_systems:.0f}")
        print(f"Number of Systems per block:       {number_of_sytems_per_block:.0f}")
        print(f"Logical Threads per block:         {logical_threads_per_block:.0f}")
        print(f"Threads per block (BlockSize):     {threads_per_block:.0f}")
        print(f"Number of block launches:          {number_of_block_launches:.0f}")
        print(f"Thread padding:                    {thread_padding:.0f}")
        print(f"Total number of blocks (GridSize): {blocks_per_grid:.0f}")
        print(f"Total number of required threads:  {total_logical_threads:.0f}")
        print(f"Thread efficiency:                 {(number_of_systems * number_of_units_per_system)/total_logical_threads:.2f}")
        print(f"Number of idle threads:            {total_logical_threads-number_of_systems*number_of_units_per_system:.0f}")

        self._check_valid_configuration()

        # --- Tolerance and Event ---
        self._abs_tol = _init_array(abs_tol, (unit_system_dimension, ), np.float64, 1e-8)
        self._rel_tol = _init_array(rel_tol, (unit_system_dimension, ), np.float64, 1e-8)
        self._event_tol = _init_array(event_tol, (max(number_of_events, 1), ), np.float64, 1e-6)
        self._event_dir = _init_array(event_dir, (max(number_of_events, 1)), np.int32, 0)

        assert len(self._abs_tol) == unit_system_dimension
        assert len(self._rel_tol) == unit_system_dimension
        assert len(self._event_tol) == max(number_of_events, 1)

        # --- Public properties with default values ---
        self.time_step_init         = init_step
        self.time_step_max          = max_step
        self.time_step_min          = min_step
        self.time_step_growth_limit = growth_limit
        self.time_step_shrink_limit = shrink_limit


        # ---- Create Data buffers  ----
        _alloc = lambda min_size, dtype, memory: {"shape": (max(min_size, 1)), "dtype": dtype, "memory": memory }
        self._data_buffers_attrs = {
            "time_domain"                : _alloc(number_of_systems * 2, np.float64, "shared"),
            "actual_time"                : _alloc(number_of_systems, np.float64, "register"),
            "actual_state"               : _alloc(unit_system_dimension * total_logical_threads, np.float64, "shared"),
            "unit_parameters"            : _alloc(number_of_unit_parameters * total_logical_threads, np.float64, "register"),
            "system_parameters"          : _alloc(number_of_systems * number_of_system_parameters, np.float64, "shared"),
            "global_parameters"          : _alloc(number_of_global_parameters, np.float64, "global"),
            "dynamic_parameters"         : _alloc(number_of_systems * number_of_dynamic_parameters, np.float64, "shared"),
            "unit_accessories"           : _alloc(total_logical_threads * number_of_unit_accessories, np.float64, "register"),
            "system_accessories"         : _alloc(self._number_of_systems * number_of_system_accessories, np.float64, "shared"),
            "dense_output_index"         : _alloc(number_of_systems, np.int32, "global"),
            "dense_output_time_instances": _alloc(number_of_systems * number_of_dense_outputs, np.float64, "global"),
            "dense_output_states"        : _alloc(total_logical_threads * unit_system_dimension * number_of_dense_outputs, np.float64, "global"),
            "status"                     : _alloc(number_of_systems, np.int8, "global"),
            "coupling_matrices"          : _alloc(self._number_of_coupling_matrices * number_of_units_per_system * number_of_units_per_system * number_of_systems, np.float64, "shared" )
        }
        self._check_memory_usage()


        self._data_buffers_host = {}
        for name, attr in self._data_buffers_attrs.items():
            self._data_buffers_host[name] = cuda.pinned_array(shape=attr["shape"], dtype=attr["dtype"])
        
        self._data_buffers_device = {
            k: cuda.device_array_like(v) for k, v in self._data_buffers_host.items()
        }

        # ---- Create the Kernel function ----
        self._cuda_jit_kwargs = CUDA_JIT_PROPERTIES.copy()
        if cuda_jit_kwargs is not None:
            self._cuda_jit_kwargs.update(**cuda_jit_kwargs)
        self._njit_cuda_kernel = _RKCK45_kernel(
                self.thread_config,
                self._number_of_systems,
                self._number_of_systems_per_block,
                self._number_of_units_per_system,
                self._unit_system_dimension,
                self._number_of_unit_paramenters,
                self._number_of_system_parameters,
                self._number_of_global_parameters,
                self._number_dynamic_parameters,
                self._number_of_unit_accessories,
                self._number_of_system_accessories,
                self._number_of_coupling_matrices,
                self._number_of_coupling_terms,
                self._number_of_coupling_factors,
                self._number_of_events,
                self._number_of_dense_outputs,
                self._method,
                self._linsolve,
                self._abs_tol,
                self._rel_tol,
                self._event_tol,
                self._event_dir,
                self._cuda_jit_kwargs
            )


        print("End")
    def solve_my_ivp(self):
        self._njit_cuda_kernel[self.thread_config["blocks_per_grid"], self.thread_config["threads_per_block"]](
            self._data_buffers_device["actual_state"],
            self._data_buffers_device["unit_parameters"],
            self._data_buffers_device["system_parameters"],
            self._data_buffers_device["global_parameters"],
            self._data_buffers_device["dynamic_parameters"],
            self._data_buffers_device["coupling_matrices"],
            self._data_buffers_device["unit_accessories"],
            self._data_buffers_device["system_accessories"],
            self._data_buffers_device["time_domain"],
            self._data_buffers_device["actual_time"],
            self._data_buffers_device["status"],
            self._data_buffers_device["dense_output_index"],
            self._data_buffers_device["dense_output_time_instances"],
            self._data_buffers_device["dense_output_states"],
            self.time_step_init,
            self.time_step_max,
            self.time_step_min,
            self.time_step_growth_limit,
            self.time_step_shrink_limit)


    def status(self):
        """
        Algorithm termination:
            -1: integration failure
             0: time_domain end is reached
            +1: termination event occured 
        """
        self.sync_to_device("status")
        return np.array(self._data_buffers_host["status"], dtype=np.int8)

    def set_host_value_coupling_matrix(self,system_id: int, coupling_id: int, unit_id: int, coupled_unit_id: int, value: float, sync: bool = False):
        assert 0 <= system_id < self._number_of_systems
        assert 0 <= coupling_id < self._number_of_coupling_matrices
        assert 0 <= unit_id < self._number_of_units_per_system
        assert 0 <= coupled_unit_id < self._number_of_units_per_system
        # Flattened index számítása:
        # coupling-major → i → j → system-major
        # idx = (((coupling_id * U + i) * U + j) * S + system_id)
        idx = (((coupling_id * self._number_of_units_per_system + unit_id) * self._number_of_units_per_system + coupled_unit_id) * self._number_of_systems) + system_id
        self._data_buffers_host["coupling_matrices"][idx] = np.float64(value)
        # layout --> [system, matrix_id, row, col]
        if sync:
            self.sync_to_device("coupling_matrices")

    def set_host_value_unit_scope(self, system_id: int, unit_id: int, parameter: str, index: int, value: float, sync: bool = False):
        """
        Set Host variable, unit scope
        """
        assert parameter in ["actual_state", "unit_parameters", "unit_accessories"], f"Err: set_host_value_unit_scope: {parameter} is not in the unit scope"
        block_id         = int(system_id // self._number_of_systems_per_block)
        local_system_id  = int(system_id % self._number_of_systems_per_block)
        
        global_thread_id = block_id * (self.thread_config["logical_threads_per_block"] + self.thread_config["thread_padding"]) \
                         + local_system_id * self._number_of_units_per_system \
                         + unit_id 
        
        idx = global_thread_id + index * self.thread_config["total_logical_threads"]        # Global memory idx
        self._data_buffers_host[parameter][idx] = np.float64(value)
        if sync:
            self._data_buffers_device[parameter][idx] = np.float64(value)

    def set_host_value_system_scope(self, system_id: int, parameter: str, index: int, value: float, sync: bool = False):
        """
        Set Host variable, system scope
        """
        assert parameter in ["time_domain", "actual_time", "system_parameters", "dynamic_parameters", "status"], f"Err: set_host_value_system_scope: {parameter} is not in the system scope"
        idx = system_id + index * self._number_of_systems
        self._data_buffers_host[parameter][idx] = np.float64(value)
        if sync:
            self._data_buffers_device[parameter][idx] = np.float64(value)

    def set_host_value_global_scope(self, parameter: str, index: int, value: float, sync: bool = False):
        """
        Set Host variable, global scope
        """
        assert parameter in ["global_parameters"], f"Err: set_host_value_global_scope: {parameter} is not in the global scope"
        self._data_buffers_host[parameter][index] = np.float64(value)
        if sync:
            self._data_buffers_device[parameter][index] = np.float64(value)

    def get_host_value_unit_scope(self, system_id: int, unit_id: int, parameter: str, index: int):
        """
        Get Host variable, unit scope
        """
        assert parameter in["actual_state", "unit_parameters", "unit_acessories"],  f"Err: get_host_value_unit_scope: {parameter} is not in the unit scope"
        block_id        = int(system_id // self._number_of_systems_per_block)
        local_system_id = int(system_id % self._number_of_systems_per_block)

        global_thread_id = block_id * (self.thread_config["logical_threads_per_block"] + self.thread_config["thread_padding"]) \
                         + local_system_id * self._number_of_units_per_system \
                         + unit_id
        
        idx = global_thread_id + index * self.thread_config["total_logical_threads"]  
        return self._data_buffers_host[parameter][idx]

    def get_host_value_system_scope(self, system_id: int, parameter: str, index: int):
        assert parameter in ["time_domain", "actual_time", "system_parameters", "dynamic_parameters", "status"], f"Err: get_host_value_system_scope: {parameter} is not in the system scope"
        idx = system_id + index * self._number_of_systems
        return self._data_buffers_host[parameter][idx]

    def get_host_value_global_scope(self, parameter: str, index: int):
        assert parameter in ["global_parameters"], f"Err: get_host_value_global_scope: {parameter} is not in the global scope"
        return self._data_buffers_host[parameter][index]

    def get_dense_output(self):
        """
        Return Dense Output as numpy arrays
        """
        dense_index = self._data_buffers_host["dense_output_index"].reshape((self._number_of_systems, ))
        dense_time  = self._data_buffers_host["dense_output_time_instances"].reshape((self._number_of_dense_outputs, self._number_of_systems))
        dense_state = self._data_buffers_host["dense_output_states"].reshape((self._number_of_dense_outputs, self._unit_system_dimension, self.thread_config["total_logical_threads"]))
        return dense_index, dense_time, dense_state

    def sync_to_device(self, parameter: str):
        """
        Copy arrays from the host buffers to the device buffers
        """
        if parameter == "all":
            for k, v in self._data_buffers_host.items():
                self._data_buffers_device[k].copy_to_device(v)
        else:
           self._data_buffers_device[parameter].copy_to_device(self._data_buffers_host[parameter]) 

    def sync_to_host(self, parameter: str):
        """
        Copy arrays from the devices buffers to the host buffers
        """
        if parameter == "all":
            for k, v in self._data_buffers_device.items():
                v.copy_to_host(self._data_buffers_host[k])
        else:
            self._data_buffers_device[parameter].copy_to_host(self._data_buffers_host[parameter])

    def _calculate_memory_requirements(self):
        memory_summary = {
            "register" : 0,
            "shared"   : 0,
            "global"   : 0,
            "total"    : 0
        }

        for name, attr in self._data_buffers_attrs.items():
            shape, dtype, memory = attr["shape"], attr["dtype"], attr["memory"]

            if memory not in memory_summary:
                raise ValueError(f"Unknown memory type '{memory}' in buffer '{name}'")
            
            num_elements = np.prod(shape, dtype=np.uint64)
            bytes_per_element = np.dtype(dtype).itemsize
            memory_summary[memory] += num_elements * bytes_per_element

        memory_summary["total"] = memory_summary["register"] + memory_summary["shared"] + memory_summary["global"]

        print("=== Memory Usage Summary ===")
        for memory_type, total_bytes in memory_summary.items():
            kb = total_bytes / 1024
            print(f"{memory_type.capitalize():<10}: {total_bytes} B ({kb:.2f} KiB)")
        return memory_summary

    def _check_memory_usage(self):
        memory_summary = self._calculate_memory_requirements()
        max_shared_memory = self._active_device.MAX_SHARED_MEMORY_PER_BLOCK
        try:
            free_memory, _ = cuda.current_context().get_memory_info()
        except cuda.CudaSupportError:
            raise RuntimeError("No active CUDA context found. Please ensure a valid GPU is selected.")

        required_memory_MB = memory_summary["total"] / (1024 ** 2)
        free_memory_MB = free_memory / (1024 ** 2)

        print(f"Required memory: {required_memory_MB:.2f} MB")
        print(f"Free GPU memory: {free_memory_MB:.2f} MB")
        print("-----------------------------------------")
        if required_memory_MB > free_memory_MB:
            raise ValueError(
                f"Error: The required memory usge ({required_memory_MB:.2f} MB) exceeds "
                f"the device's available memory ({free_memory_MB:.2f} MB)."
            )
        else:
            print(f"Global Memory check passed: The required memory fits within the available GPU memory: {required_memory_MB:.2f}/{free_memory_MB:.2f} MB")

        required_shared_memory = memory_summary["shared"] / self.thread_config["blocks_per_grid"]
        if required_shared_memory > max_shared_memory:
            raise ValueError(
                f"Error: The shared memory usage per block ({required_shared_memory / 1024:.2f} KB) exceeds "
                f"the device's maximum shared memory per block ({max_shared_memory / 1024:.2f} KB)."
            )
        else:
            print(f"Shared Memory check passed: The Shared memory usage is within limits: {required_shared_memory / 1024:.2f}/{max_shared_memory / 1024:.2f} KB")

    def _check_valid_configuration(self):
        if self.thread_config["number_of_block_launches"] != 1:
            raise ValueError(
                f"The required system configuration is not supported \n"
                f"The BLOCKSIZE must be (SPB x UPS) <= TPB \n"
                f"Try to increase the BLOCKSIZE or decrease SPB (systems per block)"
            )

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

    



# ----------  ALGO ---------------
# --------------------------------
# ----- Main Kernel Function -----
def _RKCK45_kernel(
        thread_config: Dict,
        number_of_systems: int,
        number_of_systems_per_block: int,
        number_of_units_per_system: int,
        unit_system_dimension: int,
        number_of_unit_parameters: int,
        number_of_system_parameters: int,
        number_of_global_parameters: int,
        number_of_dynamic_parameters: int,
        number_of_unit_accessories: int,
        number_of_system_accessories: int,
        number_of_coupling_matrices: int,
        number_of_coupling_terms: int,
        number_of_coupling_factors: int,
        number_of_events: int,
        number_of_dense_output: int,
        method: str,
        linalg: str,
        abs_tol: np.ndarray,
        rel_tol: np.ndarray,
        event_tol: np.ndarray,
        event_dir: np.ndarray,
        cuda_jit: Dict,
    ):

    # CONSTANT parameter ---------
    # Create aliases ---
    # Thread Config -
    LOGIC_THREADS_PER_BLOCK  = thread_config["logical_threads_per_block"]
    THREADS_PER_BLOCK        = thread_config["threads_per_block"]
    THREAD_PADDING           = thread_config["thread_padding"]
    NUM_TOTAL_THREADS        = thread_config["total_logical_threads"]
    NUM_BLOCKS               = thread_config["blocks_per_grid"]
    NUM_BLOCK_LAUNCHES       = thread_config["number_of_block_launches"]
    
    # Parameters
    NS      = number_of_systems
    SPB     = number_of_systems_per_block
    UPS     = number_of_units_per_system
    UD      = unit_system_dimension
    NUP     = number_of_unit_parameters
    NSP     = number_of_system_parameters
    NGP     = number_of_global_parameters
    NDP     = number_of_dynamic_parameters
    NUA     = number_of_unit_accessories
    NSA     = number_of_system_accessories
    NC      = number_of_coupling_matrices
    NCT     = number_of_coupling_terms
    NCF     = number_of_coupling_factors
    NE      = number_of_events
    NDO     = number_of_dense_output
    ALGO    = method
    LIN_ALG = linalg
    ATOL    = abs_tol
    RTOL    = rel_tol
    ETOL    = event_tol
    EDIR    = event_dir
    CUDA_JIT = cuda_jit

    # Padding ---- ??
    PADDING = (0, 4, 8, 16)
    NSP_P   = NSP + 1 if NSP in PADDING else NSP
    NGP_P   = NGP + 1 if NGP in PADDING else NGP
    NDP_P   = NDP + 1 if NDP in PADDING else NDP
    NSA_P   = NSA + 1 if NSA in PADDING else NSA
    UPS_P   = UPS + 1 if UPS in PADDING else UPS


    # Flag Names 
    IS_TERMINATED   = 0
    USER_TERMINATED = 1
    END_TIME_DOMAIN = 2
    IS_FINITE       = 3
    UPDATE_STEP     = 4
    EVENT_TERMINATED= 5


    # DEVICE FUNCTIONS CALLED IN THE KERNEL
    # -------------------------------------
    @cuda.jit(nb.void(
            nb.int32,
            nb.int32,
            nb.int32,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.int32[:,::]
    ), device=True, inline=True)
    def block_error_control_RKCK45(
        gsid,       # Global System ID
        lsid,       # Local System ID,
        luid,       # Local Unit ID,
        l_next_state,
        l_actual_state,
        l_error,
        s_time_step,
        s_new_time_step,
        time_step_max,
        time_step_min,
        time_step_growth_limit,
        time_step_shrink_limit,
        s_flags
    ):
        s_relative_error = cuda.shared.array((SPB, ), dtype=nb.float64)
        lid = cuda.threadIdx.x
        if ((lid < SPB) and s_flags[lid, IS_TERMINATED] == 0):
            # Initialize non terminated systems
            s_relative_error[lid]     = 1.0e30
            s_flags[lid][UPDATE_STEP] = 1
            s_flags[lid][IS_FINITE]   = 1
        cuda.syncthreads()
        
        # - Error Calculation -
        if s_flags[lsid, IS_TERMINATED] == 0:
            l_relative_error = 1.0e30
            l_update_step = True
            for i in range(UD):
                error_tolerance  =  max( RTOL[i]*max( abs(l_next_state[i]), abs(l_actual_state[i])), ATOL[i] )
                l_update_step    = l_update_step and ( l_error[i] < error_tolerance )
                l_relative_error = min(l_relative_error, error_tolerance / l_error[i])

            # - Check finite error -
            if math.isfinite(l_relative_error):
                cuda.atomic.min(s_relative_error, lsid, l_relative_error)
            else:
                cuda.atomic.and_(s_flags[lsid], IS_FINITE, 0)
                print("Error Calculation: Global System ID: ", gsid, " State is not finite")
            if l_update_step == 0:
                cuda.atomic.and_(s_flags[lsid], UPDATE_STEP, 0)
        cuda.syncthreads()

        # - New time step -
        if ((lid < SPB) and s_flags[lid, IS_TERMINATED] == 0):
            gid = lid + cuda.blockIdx.x
            power = 0.2 if s_flags[lsid, UPDATE_STEP] == 1 else 0.5
            time_step_multiplicator = 0.9 * math.pow(s_relative_error[lsid], power)

            # - Check finite time_step  -
            if math.isfinite(time_step_multiplicator) == False:
                print("TimeStep Calculation: Global System ID: ", gid, " State is not finite")
                s_flags[lid, IS_FINITE] = 0

            if s_flags[lid, IS_FINITE] == 0:
                # Infinite state --> Reduce the timestep
                time_step_multiplicator = time_step_shrink_limit
                s_flags[lid, UPDATE_STEP] = 0

                if s_time_step[lid] < time_step_min*1.01:
                    s_flags[lid, IS_TERMINATED] = 1
                    print("TimeStep Reduction: Global System ID: ",gid," State is not a finite. The minimum step size is reached")
            else:
                # Finite state --> Check minimum step size is reached --> terminate the system to avoid crash
                if s_time_step[lid] < time_step_min*1.01:
                    print("TimeStep Reduction: Global System ID: ",gid," The minimum step size is reached")
                    s_flags[lid, UPDATE_STEP]   = 0
                    s_flags[lid, IS_TERMINATED] = 1

            # - Finally, set the new time step value -
            # - Clip the miltiplicator -
            time_step_multiplicator = min(time_step_multiplicator, time_step_growth_limit)
            time_step_multiplicator = max(time_step_multiplicator, time_step_shrink_limit)

            # - Clip the time step -
            new_time_step = s_time_step[lid] * time_step_multiplicator
            new_time_step = min(new_time_step, time_step_max)
            new_time_step = max(new_time_step, time_step_min)

            s_new_time_step[lid] = new_time_step

        cuda.syncthreads()

    # --- DENSE OUTPUT ---
    @cuda.jit(nb.void(
            nb.int32,
            nb.int32,
            nb.int32,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.uint32[:],
            nb.float64,
            nb.float64,
    ), device=True, inline=True)
    def block_store_dense_output(
            gtid,
            lsid,
            luid,
            dense_output_time_instances,
            dense_output_states,
            l_actual_state,
            s_actual_time,
            s_dense_output_time,
            s_dense_output_time_index,
            l_dense_output_min_time_step,
            time_domain_end):
        
        update_dense_states = cuda.shared.array((SPB, ), dtype=nb.boolean)
        lid = cuda.threadIdx.x
        gid = lid + cuda.blockIdx.x * SPB        # Global System ID
        if ((lid < SPB) and 
            (s_dense_output_time[lid] < NDO) and 
            (s_dense_output_time[lid] <= s_actual_time[lid])):
            # Check Store Condition
            update_dense_states[lid] = True
            
            # Store dense time instances 
            dense_output_time_instances[gid + s_dense_output_time_index[lid] * NS] = s_actual_time[lid]
            
            # Update for next save time
            s_dense_output_time_index[lid] +=1
            s_dense_output_time[lid] = min(s_actual_time[lid] + l_dense_output_min_time_step, time_domain_end)

        else:
            update_dense_states[lid] = False
        cuda.syncthreads()

        if update_dense_states[lsid]:
            t_idx = s_dense_output_time_index[lsid] - 1 
            dense_output_state_index = gtid + t_idx * NUM_TOTAL_THREADS * UD
            for i in range(UD):
                dense_output_states[dense_output_state_index] = l_actual_state[i]
                dense_output_state_index += NUM_TOTAL_THREADS

        cuda.syncthreads()


    # --- STEPPER FUNCTIONS ---
    @cuda.jit(
        nb.void(
            nb.int32,
            nb.int32,
            nb.int32,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:,::],
            nb.float64[:,::],
            nb.float64[:],
            nb.float64[:,::],
            nb.float64[:],
            nb.float64[:],
            nb.int32[:,::]
    ), device=True, inline=True)
    def block_stepper_RKCK45(
        gsid,            # Global System ID
        lsid,            # Local  System ID
        luid,            # Local Unit ID
        l_next_state,
        l_actual_state,
        l_error,
        l_coupling_factor,
        l_unit_parameters,
        s_global_parameters,
        s_system_parameters,
        s_dynamic_parameres,
        l_unit_accessories,
        s_sytem_accessories,
        s_actual_time,
        s_time_step,
        s_flags):
        
        # TODO: INLCUDE ACCESSORIES, AND COUPLINGS

        # RKCK STAGES
        k1 = cuda.local.array((UD, ), dtype=nb.float64)
        k2 = cuda.local.array((UD, ), dtype=nb.float64)
        k3 = cuda.local.array((UD, ), dtype=nb.float64)
        k4 = cuda.local.array((UD, ), dtype=nb.float64)
        k5 = cuda.local.array((UD, ), dtype=nb.float64)
        k6 = cuda.local.array((UD, ), dtype=nb.float64)
        x  = cuda.local.array((UD, ), dtype=nb.float64)
        t0 = s_actual_time[lsid]
        t  = s_actual_time[lsid]
        dt = s_time_step[lsid]

        # --- K1 ---
        per_block_ode_function(
            gsid, luid, t, 
            k1, l_actual_state, l_coupling_factor,
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()
        #per_block_explicit_coupling_correction()
        #instantaneous coupling matrix + linsolve

        # --- K2 ---
        t = t0 + dt * (1.0 / 5.0)
        for i in range(UD):
            x[i] = l_actual_state[i] + dt * (1.0 / 5.0) * k1[i]
        
        per_block_ode_function(
            gsid, luid, t, 
            k2, x, l_coupling_factor,
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()

        # --- K3 ---
        t = t0 + dt * (3.0 / 10.0)
        for i in range(UD):
            x[i] = l_actual_state[i] \
                    + dt * (  (3.0 / 40.0) * k1[i] \
                            + (9.0 / 40.0) * k2[i] )
            
        per_block_ode_function(
            gsid, luid, t, 
            k3, x, l_coupling_factor,
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()

        # --- K4 ---
        t = t0 + dt * (3.0 / 5.0)
        for i in range(UD):
            x[i] = l_actual_state[i] \
                    + dt * (  (3.0 / 10.0) * k1[i] \
                            - (9.0 / 10.0) * k2[i] \
                            + (6.0 / 5.0)  * k3[i] )
        
        per_block_ode_function(
            gsid, luid, t, 
            k4, x, l_coupling_factor, 
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()

        # --- K5 ---
        t = t0 + dt
        for i in range(UD):
            x[i] = l_actual_state[i] \
                    + dt * (- (11.0 / 54.0) * k1[i] \
                            + (5.0 / 2.0 )  * k2[i] \
                            - (70.0 / 27.0) * k3[i] \
                            + (35.0 / 27.0) * k4[i] )
        
        per_block_ode_function(
            gsid, luid, t, 
            k5, x, l_coupling_factor,
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()

        # --- K6 ---
        t = t0 + dt * (7.0 / 8.0)
        for i in range(UD):
            x[i] = l_actual_state[i] \
                    + dt * ( (1631.0/55296.0) * k1[i] \
                            + (175.0/512.0)   * k2[i] \
                            + (575.0/13824.0) * k3[i] \
                            + (44275.0/110592.0) * k4[i] \
                            + (253.0/4096.0)  * k5[i] )
        
        per_block_ode_function(
            gsid, luid, t, 
            k6, x, l_coupling_factor,
            l_unit_parameters, s_global_parameters,
            s_system_parameters[lsid], s_dynamic_parameres[lsid])
        cuda.syncthreads()

        # --- NEW STATE AND ERROR ---
        for i in range(UD):
            l_next_state[i] = l_actual_state[i] \
                            + dt * ( ( 37.0/378.0)  * k1[i] \
                                    + (250.0/621.0) * k3[i] \
                                    + (125.0/594.0) * k4[i] \
                                    + (512.0/1771.0)* k6[i] )

            l_error[i] =  (  37.0/378.0  -  2825.0/27648.0 ) * k1[i] \
                        + ( 250.0/621.0  - 18575.0/48384.0 ) * k3[i] \
                        + ( 125.0/594.0  - 13525.0/55296.0 ) * k4[i] \
                        + (   0.0        -   277.0/14336.0 ) * k5[i] \
                        + ( 512.0/1771.0 -     1.0/4.0     ) * k6[i]

            l_error[i] = dt * abs( l_error[i] ) + 1e-18
            
            if (math.isfinite(l_next_state[i]) == False or math.isfinite(l_error[i]) == False ):
                s_flags[lsid, IS_FINITE] = 0
        cuda.syncthreads()



    # -------------------------------------
    # --------- KERNEL FUNCTION -----------
    # -------------------------------------
    @cuda.jit(**CUDA_JIT)
    def coupled_system_per_block(
        actual_state,
        unit_parameters,
        system_parameters,
        global_parameters,
        dynamic_parameters,
        coupling_matrices,
        unit_accessories,
        system_accessories,
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
        # Multiple Systems + Single Block Launch!!

        # -- THREAD MANAGEMENT --
        lsid = nb.int32(cuda.threadIdx.x // UPS)                               # Local System ID
        luid = nb.int32(cuda.threadIdx.x % UPS)                                # Local Unit ID
        gsid = nb.int32(lsid + cuda.blockIdx.x * SPB)                          # Global System ID
        gtid = nb.int32(cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x)    # Global Thread ID

        # -- SHARED MEMORY MANAGEMENT --
        # TODO: Check Bank conflict!!!
        # - INITIALIZE SYSTEM SCOPE VARIABLES -
        # Time Domain
        s_time_domain = cuda.shared.array((SPB, 2), dtype=nb.float64)
        s_actual_time = cuda.shared.array((SPB, ),  dtype=nb.float64)
        s_time_step   = cuda.shared.array((SPB, ),  dtype=nb.float64)
        s_new_time_step = cuda.shared.array((SPB, ),dtype=nb.float64)

        # Shared Parameters
        s_global_parameters = cuda.shared.array((NGP_P, ), dtype=nb.float64)                # Hopefully fits into the smem
        s_system_parameters = cuda.shared.array((SPB, NSP_P), dtype=nb.float64)
        s_dynamic_parameters = cuda.shared.array((SPB, NDP_P), dtype=nb.float64)
        s_system_accessories = cuda.shared.array((SPB, NSA_P), dtype=nb.float64)

        #s_coupling_matrices = cuda.shared.array((SPB, NC, UPS, UPS_P), dtype=nb.float64)   # Figure out the shape!

        # Dense Output variables
        if NDO > 0:
            s_dense_output_time_index = cuda.shared.array((SPB, ), dtype=nb.uint32)
            s_dense_output_time       = cuda.shared.array((SPB, ), dtype=nb.float64)

        # Runtime Flags
        s_terminated_systems_per_block = cuda.shared.array((1, ), dtype=nb.int32)         # cuda.atomic.add( )  32bit!
        s_flags = cuda.shared.array((SPB, 6), dtype=nb.int32)

        if cuda.threadIdx.x == 0:
            s_terminated_systems_per_block[0] = 0
        cuda.syncthreads()

        # - LOAD SYSTEM SCOPE VARIABLES TO SHARED MEMORY -
        lid = cuda.threadIdx.x                   # Local System ID
        gid = lid + cuda.blockIdx.x * SPB        # Global System ID

        if (lid < SPB) and (gid < NS):
            # - Simulation Time -
            for i in range(2):
                s_time_domain[lid, i] = time_domain[gid + i * NS]
                if i == 0:
                    s_actual_time[lid]   = time_domain[gid]
                    s_time_step[lid]     = time_step_init
                    s_new_time_step[lid] = time_step_init

            # - Shared Parameters -
            for i in range(NSP):
                s_system_parameters[lid, i] = system_parameters[gid + i * NS]

            # - Dynamic Shared Parameters -
            for i in range(NDP):
                s_dynamic_parameters[lid, i] = dynamic_parameters[gid + i * NS]

            # - Global Shared Parameters -
            if lid == 0:
                for i in range(NGP):
                    s_global_parameters[i] = global_parameters[i]                       # No need for i * NS

            # - System Accessories -
            for i in range(NSA):
                s_system_accessories[lid, i] = system_accessories[gid + i * NS]
            
            # - Coupling Matrices -
            # TODO: Figure out memory layout

            # - Initialize Flags
            s_flags[lid, IS_TERMINATED]   = 0
            s_flags[lid, USER_TERMINATED] = 0        
            s_flags[lid, END_TIME_DOMAIN] = 0
            if NDO > 0:
                s_dense_output_time_index[lid] = 0
                s_dense_output_time[lid]       = s_actual_time[lid]

        elif (lid < SPB) and (gid >= NS):
            # Not a valid system!
            cuda.atomic.add(s_terminated_systems_per_block, 0, 1)
            # Set flags to terminated status
            s_flags[lid, IS_TERMINATED] = 1
            s_flags[lid, UPDATE_STEP]   = 0
            # TODO: Dense output?
        cuda.syncthreads()      
        
        # - LOAD UNIT SCOPE VARIABLES - 
        l_actual_state = cuda.local.array((UD, ), dtype=nb.float64)
        l_next_state   = cuda.local.array((UD, ), dtype=nb.float64)
        l_error        = cuda.local.array((UD, ), dtype=nb.float64)
        l_unit_parameters = cuda.local.array((NUP, ) if NUP !=0 else (1, ), dtype=nb.float64)
        l_unit_accessories = cuda.local.array((NUA, )if NUA !=0 else (1, ), dtype=nb.float64)

        l_coupling_factors = cuda.local.array((NCF, ), dtype=nb.float64)

        if gtid < NUM_TOTAL_THREADS:
            for i in range(UD):
                l_actual_state[i] = actual_state[gtid + i * NUM_TOTAL_THREADS]

            for i in range(NUP):
                l_unit_parameters[i] = unit_parameters[gtid + i * NUM_TOTAL_THREADS]

            for i in range(NUA):
                l_unit_accessories[i] = unit_accessories[gtid + i * NUM_TOTAL_THREADS]

        cuda.syncthreads()
        
        if gsid < NS:
            # - SOLVER INITIALIZATION -
            if (s_flags[lsid][IS_TERMINATED] == 0):

                if (NDO > 0):
                    l_dense_output_min_time_step = (s_time_domain[lsid, 1] - s_time_domain[lsid, 0]) / (NDO - 1)
                    block_store_dense_output(
                        gtid,
                        lsid,
                        luid,
                        dense_output_time_instances,
                        dense_output_states,
                        l_actual_state,
                        s_actual_time,
                        s_dense_output_time,
                        s_dense_output_time_index,
                        l_dense_output_min_time_step,
                        s_time_domain[lsid, 1],
                    )
                    print(l_dense_output_min_time_step)
                cuda.syncthreads()

                # - INITIALIZATION -
                per_block_initialization(
                    gsid,
                    luid,
                    s_actual_time[lsid],
                    s_time_domain[lsid],
                    l_actual_state,
                    l_coupling_factors,
                    l_unit_parameters,
                    s_global_parameters,
                    s_system_parameters[lsid],
                    s_dynamic_parameters[lsid])
                cuda.syncthreads()

                if (NE > 0):
                    # per_block_event_function
                    # per_block_event_time_step_control
                    pass
                cuda.syncthreads()

            # - SOLVER MAIN LOOP -
            while (s_terminated_systems_per_block[0] < SPB):
                # INITIALIZE TIME STEP
                lid = cuda.threadIdx.x
                if ((lid < SPB) and s_flags[lid, IS_TERMINATED] == 0):
                    s_flags[lid, UPDATE_STEP] = 1
                    s_flags[lid, IS_FINITE]   = 1

                    if (s_new_time_step[lid] > (s_time_domain[lid][1] - s_actual_time[lid])):
                        s_time_step[lid] = s_time_domain[lid][1] - s_actual_time[lid]
                        s_flags[lid, END_TIME_DOMAIN] = 1
                    else:
                        s_time_step[lid] = s_new_time_step[lid]
                        s_flags[lid, END_TIME_DOMAIN] = 0

                cuda.syncthreads()

                # FORWARD STEPPING
                if ALGO == "RKCK45":
                    block_stepper_RKCK45(
                        gsid,            # Global System ID
                        lsid,            # Local  System ID
                        luid,            # Local Unit ID
                        l_next_state,
                        l_actual_state,
                        l_error,
                        l_coupling_factors,
                        l_unit_parameters,
                        s_global_parameters,
                        s_system_parameters,
                        s_dynamic_parameters,
                        l_unit_accessories,
                        s_system_accessories,
                        s_actual_time,
                        s_time_step,
                        s_flags)
                    
                    block_error_control_RKCK45(
                        gsid,
                        lsid,
                        luid,
                        l_next_state,
                        l_actual_state,
                        l_error,
                        s_time_step,
                        s_new_time_step,
                        time_step_max,
                        time_step_min,
                        time_step_growth_limit,
                        time_step_shrink_limit,
                        s_flags
                    )


                # - NEW EVENT VALUE AND TIME STEP CONTROL -
                if NE > 0:
                    # per block event function
                    # per block event time_step_control
                    pass


                # - SUCCESFULL TIME STEPPING UPDATE TIME AND STATE -
                if (s_flags[lsid, UPDATE_STEP] == 1):
                    if luid == 0:
                        s_actual_time[lsid] += s_time_step[lsid]
                        #if gtid == 0:
                        #    print(s_actual_time[lsid])

                    for i in range(UD):
                        l_actual_state[i] = l_next_state[i]
                        #if gtid == 0 and i == 0:
                        #    print(l_coupling_factors[3])

                    cuda.syncthreads()

                    # USER DEFINED ACTION AFTER SUCCESSFUL TIMESTEP
                    # EVENT TERMINAL
                    # EVENT FUNCION

                    if NDO > 0:
                        # STORE DENSE OUTPUT
                        block_store_dense_output(
                            gtid,
                            lsid,
                            luid,
                            dense_output_time_instances,
                            dense_output_states,
                            l_actual_state,
                            s_actual_time,
                            s_dense_output_time,
                            s_dense_output_time_index,
                            l_dense_output_min_time_step,
                            s_time_domain[lsid][1]
                        )

                # - CHECK TERMINATION -
                lid = cuda.threadIdx.x
                if lid < SPB:
                    if (s_flags[lid, IS_TERMINATED] == 1 or
                        s_flags[lid, USER_TERMINATED] == 1 or
                        s_flags[lid, END_TIME_DOMAIN] == 1 or
                        s_flags[lid, EVENT_TERMINATED] == 1):

                        # Count only the new terminal events
                        if s_flags[lid, IS_TERMINATED] == 0:
                            s_flags[lid, IS_TERMINATED] = 1
                            cuda.atomic.add(s_terminated_systems_per_block, 0, 1)

                cuda.syncthreads()
                 
                # Avoid endless loop during implementation --> increas terminated system coutn after every iteration
                #if cuda.threadIdx.x == 0:
                #    cuda.atomic.add(s_terminated_systems_per_block, 0, 1)

            # - FINALIZATION -
            per_block_finalization(
                gsid,
                luid,
                s_actual_time[lsid],
                s_time_domain[lsid],
                l_actual_state,
                l_coupling_factors,
                l_unit_parameters,
                s_global_parameters,
                s_system_parameters[lsid],
                s_dynamic_parameters[lsid]
            )
            cuda.syncthreads()
        # - SOLVER TERMINATED, COPY DATA TO GLOBAL MEMORY -

        # - UNIT SCOPE -
        if gtid < NUM_TOTAL_THREADS:
            for i in range(UD):
                actual_state[gtid + i * NUM_TOTAL_THREADS] = l_actual_state[i]

            for i in range(NUP):
                unit_parameters[gtid + i * NUM_TOTAL_THREADS] = l_unit_parameters[i]

            for i in range(NUA):
                unit_accessories[gtid + i * NUM_TOTAL_THREADS] = l_unit_accessories[i]

        cuda.syncthreads()


        # - SYSTEM SCOPE -
        if (lid < SPB) and (gid < NS):
            if NDO > 0:
                dense_output_index[gid] = s_dense_output_time_index[lid]


        cuda.syncthreads()

        # DEBUG LOG
        if gtid == 0:
            print("gtid: ",gtid, "gsid: ", gsid)
            #print("Logical Threads per block:", LOGIC_THREADS_PER_BLOCK)
            #print("Threads per block (BlockSize):", THREADS_PER_BLOCK)
            #print("Thread padding:", THREAD_PADDING)
            #print("Total number of required threads:", NUM_TOTAL_THREADS)
            #print("Total number of blocks (GridSize):", NUM_BLOCKS)
            #print("Number of block launches:", NUM_BLOCK_LAUNCHES)
            #print(tid, s_terminated_systems_per_block[0])
            print(gtid, l_actual_state[0])
            #print(gtid, s_dynamic_parameters[0, 0])
            #print(gtid, s_dynamic_parameters[0, 1])
            #print(gtid, s_dynamic_parameters[0][2])
            #print(gtid, s_dynamic_parameters[0][3])
            #print(gtid, s_dynamic_parameters[0][4])
            #print(gtid, s_dynamic_parameters[0, 5])
            #print(gtid, s_global_parameters[0])
            #print(gtid, s_global_parameters[1])
            #print(gtid, s_global_parameters[2])
            #print(gtid, s_global_parameters[3])
            #print(gtid, s_global_parameters[4])
            #print(gtid, s_flags[0, IS_TERMINATED])
            #print(gtid, s_system_accessories[0][0])
            #print(gtid, l_unit_parameters[0])
            #print(gtid, l_unit_parameters[1])
            #print(gtid, l_unit_parameters[2])
            #print(gtid, l_unit_parameters[3])
            #print(gtid, l_unit_parameters[4])
            #print(gtid, l_unit_parameters[5])
            #print(gtid, l_unit_parameters[6])
            #print(gtid, l_unit_parameters[7])
            #print(gtid, l_unit_parameters[8])
            #print(gtid, l_unit_parameters[9])
            #print(gtid, l_unit_parameters[10])
            #print(gtid, l_unit_parameters[11])
            #print(gtid, l_unit_parameters[12])








    return coupled_system_per_block

