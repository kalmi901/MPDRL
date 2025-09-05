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
                          "per_block_explicit_coupling",
                          "per_block_implicit_mat_vec",
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
            #print(func_name)
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
                lin_abs_tol : float = 1e-9,
                lin_rel_tol:  float = 1e-9,
                lin_max_iter: int = 128,
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
        self._lin_abs_tol = lin_abs_tol
        self._lin_rel_tol = lin_rel_tol
        self._lin_maxiter = lin_max_iter
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
                self._lin_abs_tol,
                self._lin_rel_tol,
                self._lin_maxiter,
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
        
        cuda.synchronize()


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
        # layout: [system, matrix_id, row, col]  (C-order)
        idx = (((system_id * self._number_of_coupling_matrices + coupling_id) *  self._number_of_units_per_system + unit_id) *  self._number_of_units_per_system) + coupled_unit_id
        self._data_buffers_host["coupling_matrices"][idx] = np.float64(value)
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
        lin_abs_tol: float,
        lin_rel_tol: float,
        lin_maxiter: int,
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
    CDIM    = UPS * UD // 2                   # Implicit Coupling matrix dimension
    CPS     = UD // 2                         # Coupling per system
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
    LATOL   = lin_abs_tol
    LRTOL   = lin_rel_tol
    MAXITER = lin_maxiter
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
            nb.float64[:,:],
            nb.float64[:,:,:],
            nb.int32[:,::]
    ), device=True, inline=True)
    def per_block_bicgstab(
        gsid,    # Global System ID
        lsid,    # Local System ID
        luid,    # Local Unit ID
        dx,      # Derivatives (k1, k2,... k5)
        cpf,     # Coupling factors
        cpt,     # Coupling termns
        mx,       # Coupling matrices,
        s_flags
        ):
        
        x  = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)
        r    = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)
        rhat = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)
        p    = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)
        s    = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)
        Av   = cuda.shared.array((SPB, CPS, UPS), dtype=nb.float64)

        dot_tmp  = cuda.shared.array((SPB, UPS), dtype=nb.float64)
        dot_tmp2 = cuda.shared.array((SPB, UPS), dtype=nb.float64)

        # Per-System scalaras
        s_tol   = cuda.shared.array((SPB, ), dtype=nb.float64)
        s_lin_terminated = cuda.shared.array((SPB, ), dtype=nb.int32)
        s_num_iters = cuda.shared.array((SPB, ), dtype=nb.int32)
        s_num_lin_terminated = cuda.shared.array((1, ), dtype=nb.int32)
        s_alpha = cuda.shared.array((SPB, ), dtype=nb.float64)
        s_beta = cuda.shared.array((SPB, ), dtype=nb.float64)
        s_omega = cuda.shared.array((SPB, ), dtype=nb.float64)
        s_rho  = cuda.shared.array((SPB, ), dtype=nb.float64)

        eps2 = 1.0e-30

        gsid_valid = (gsid < NS)
        sys_leader = (lsid < SPB) and (luid == 0)
        sys_active = (lsid < SPB) and gsid_valid and (s_flags[lsid, IS_TERMINATED] == 0)

        # 0) Initialize Counters ------------
        if cuda.threadIdx.x == 0:
            s_num_lin_terminated[0] = 0
        cuda.syncthreads()

        # --- count inactive systems ---
        # - 1) system is out of the valid range
        # - 2) system has already terminated
        if sys_leader:
            if (not gsid_valid) or s_flags[lsid, IS_TERMINATED] == 1:
                s_lin_terminated[lsid] = 1
                s_num_iters[lsid]  = -1
                cuda.atomic.add(s_num_iters, 0, 1)
            else:
                s_lin_terminated[lsid] = 0
                s_num_iters[lsid]  = 0
        cuda.syncthreads()

        # ---------------------------------
        #  1) Initialize the system 
        #  x0 := dx[coupled_idx], 
        #  b:=x0, ||b||
        active = sys_active and s_lin_terminated[lsid] == 0
        acc = 0.0
        if active:
            for j in range(CPS):        # coupled unit dimension 
                # TODO: introduce the coupling index
                # value = dx[cp_idx[j]]
                value = dx[CPS + j]     
                x[lsid, j, luid]    = value
                acc += value * value    # ||b||^2 since b := x0
            dot_tmp[lsid, luid] = acc
        else:
            dot_tmp[lsid, luid] = 0.0
        cuda.syncthreads()

        if sys_leader and active:
            b2 = 0.0
            for i in range(UPS):
                b2 += dot_tmp[lsid, i]
            bnorm = math.sqrt(b2)
            s_tol[lsid] = max (LATOL, LRTOL * bnorm)
            s_rho[lsid] = 1.0
            s_alpha[lsid]   = 1.0
            s_omega[lsid]   = 1.0
        cuda.syncthreads()


        # -----------------------------------------
        # 2) Initialize the residuum vectors
        # r = b - Ax; 
        # rhat = r 
        # p = r 
        # ||r||^2
        if active:
            per_block_implicit_mat_vec(UPS, gsid, luid, cpf, cpt, mx, x[lsid], Av[lsid])
        cuda.syncthreads()
        
        if active:
            acc = 0.0
            for j in range(CPS):
                rj = x[lsid, j, luid] - Av[lsid, j, luid]       #b := x0
                r[lsid, j, luid] = rj
                p[lsid, j, luid] = rj
                rhat[lsid, j, luid] = rj
                acc += rj*rj
            dot_tmp[lsid, luid] = acc   # r**2
        else:
            dot_tmp[lsid, luid] = 0.0
        cuda.syncthreads()

        if sys_leader and active:
            r2 = 0.0
            for i in range(UPS):
                r2 += dot_tmp[lsid, i]
            s_rho[lsid] = r2            # since rhat = r0, (rhat @ r0) = ||r||^2
            if ( math.sqrt(r2) < s_tol[lsid]) or r2 < eps2:
                # Terminated or BiCG breakdown
                # Increase the number of solved systems
                s_lin_terminated[lsid] = 1
                s_num_iters[lsid]  = 0
                cuda.atomic.add(s_num_lin_terminated, 0, 1)
        cuda.syncthreads()

        # --- debug (óvatosan a printekkel) ---
        # TODO: delete when ready
        #if (gsid == 2) and (luid == 0):
        #    print("x(0,0)=", x[lsid, 0, 0], " X(0,1)=", x[lsid, 0, 1], " x(1,0)=", x[lsid, 1, 0], " x(1,1)=", x[lsid, 1, 1])
        #    print("Av(0,0)=", Av[lsid, 0, 0], " Av(0,1)=", Av[lsid, 0, 1], " Av(1,0)=", Av[lsid, 1, 0], " Av(1,1)=", Av[lsid, 1, 1])
        #    print("s_num_lin_solved", s_num_lin_terminated[0])
        #cuda.syncthreads()

        # 4) ---------- ITERATIION ---------------
        # 4/1) Mat-Vec, v = (A @ p)
        if sys_active and (s_lin_terminated[lsid] == 0):
            per_block_implicit_mat_vec(UPS, gsid, luid, cpf, cpt, mx, p[lsid], Av[lsid])        # v = (A @ p)
        cuda.syncthreads()
        while s_num_lin_terminated[0] < SPB:
            
            # 4/2) denom = (rhat @ v)
            if sys_active and (s_lin_terminated[lsid] == 0):
                acc = 0.0
                for j in range(CPS):
                    acc += rhat[lsid, j, luid] * Av[lsid, j, luid]
                dot_tmp[lsid, luid] = acc
            else:
                dot_tmp[lsid, luid] = 0.0
            cuda.syncthreads()

            if sys_active and (s_lin_terminated[lsid] == 0):
                denom = 0.0
                for i in range(UPS):
                    denom += dot_tmp[lsid, i]
                    
                if not math.isfinite(denom) or denom < eps2:
                    # BiCGSTAB Breakdown (rhat @ v) ~ 0
                    s_lin_terminated[lsid] = 1
                    cuda.atomic.add(s_num_lin_terminated, 0, 1)
                else:
                    s_alpha[lsid] = s_rho[lsid] / denom
            cuda.syncthreads()

            # --- Debug Print, TODO: Remove
            #if (gsid == 2) and (luid == 0):
            #    print("r(0,0)=", r[lsid, 0, 0], " r(0,1)=", r[lsid, 0, 1], " r(1,0)=", r[lsid, 1, 0], " r(1,1)=", r[lsid, 1, 1])
            #    print("v(0,0)=", Av[lsid, 0, 0], " v(0,1)=", Av[lsid, 0, 1], " v(1,0)=", Av[lsid, 1, 0], " v(1,1)=", Av[lsid, 1, 1])
            #    print("s_alpha=", s_alpha[lsid])
            #cuda.syncthreads()

            # 4/3) Half-step
            # s = r - alpha * v, ||s||^2
            if sys_active and s_lin_terminated[lsid] == 0:
                acc = 0.0
                alpha = s_alpha[lsid]
                for j in range(CPS):
                    sj = r[lsid, j, luid] - alpha * Av[lsid, j, luid]       # Av = v
                    s[lsid, j, luid] = sj
                    acc += sj * sj          # ||s||^2
                dot_tmp[lsid, luid] = acc
            else:
                dot_tmp[lsid, luid] = 0.0
            cuda.syncthreads()

            if sys_leader and s_lin_terminated[lsid] == 0:
                s2 = 0.0
                for i in range(UPS):
                    s2 += dot_tmp[lsid, i]
                if s2 < s_tol[lsid]*s_tol[lsid]:
                    # Converged half step --> 
                    # x+= alpha * p
                    s_lin_terminated[lsid] = 2
                    cuda.atomic.add(s_num_lin_terminated, 0, 1)
            cuda.syncthreads()

            # Half-step update
            # x = x + alpha * p
            if sys_active and s_lin_terminated[lsid] == 2:
                # Update x if converged during the half-step
                alpha = s_alpha[lsid]
                for j in range(CPS):
                    x[lsid, j, luid] += alpha *  p[lsid, j, luid]
                if luid == 0:
                    s_lin_terminated[lsid] = 1  # Set flag to solved!!
            cuda.syncthreads()
            
            # 4/4) mat-vec
            # t = (A @ s)
            if sys_active and s_lin_terminated[lsid] == 0:
                per_block_implicit_mat_vec(UPS, gsid, luid, cpf, cpt, mx, s[lsid], Av[lsid])        # Av = t
            cuda.syncthreads()

            #if (gsid == 2) and (luid == 0):
            #    print("s(0,0)=", s[lsid, 0, 0], " s(0,1)=", s[lsid, 0, 1], " s(1,0)=", s[lsid, 1, 0], " s(1,1)=", s[lsid, 1, 1])
            #    print("t(0,0)=", Av[lsid, 0, 0], " t(0,1)=", Av[lsid, 0, 1], " s(1,0)=", Av[lsid, 1, 0], " s(1,1)=", Av[lsid, 1, 1])
            #cuda.syncthreads()

            # 4/5) Scalar Vector products
            # tt = (t @ t)
            # ts = (t @ s)
            if sys_active and s_lin_terminated[lsid] == 0:
                acc_tt = 0.0
                acc_ts = 0.0
                for j in range(CPS):
                    tj = Av[lsid, j, luid]
                    sj = s[lsid, j, luid]
                    acc_tt += tj * tj
                    acc_ts += tj * sj
                dot_tmp[lsid, luid] = acc_tt    # ||t||^2
                dot_tmp2[lsid, luid] = acc_ts   # (t @ s)
            else:
                dot_tmp[lsid, luid] = 0.0 
                dot_tmp2[lsid, luid] = 0.0
            cuda.syncthreads()

            if sys_leader and s_lin_terminated[lsid] == 0:
                tt = 0.0
                ts = 0.0
                for i in range(UPS):
                    tt += dot_tmp[lsid, i]
                    ts += dot_tmp2[lsid, i]
                if not math.isfinite(tt) or not math.isfinite(ts) or tt < eps2:
                    # BiCGSTAB Breakdown (t @ t) ~ 0 or not finite
                    s_lin_terminated[lsid] = 1
                    cuda.atomic.add(s_num_lin_terminated, 0, 1)
                else:
                    s_omega[lsid] = ts / tt
            cuda.syncthreads()

            # 4/6) Full step update 
            # x = x + alpha * p + omega * s
            # r = s - omega * t
            # (r @ r)
            # (rhat @ r)
            if sys_active and s_lin_terminated[lsid] == 0:
                alpha = s_alpha[lsid]
                omega = s_omega[lsid]
                r2 = 0.0
                rhat2 = 0.0
                for j in range(CPS):
                    rj = s[lsid, j, luid] - omega * Av[lsid, j, luid]
                    x[lsid, j, luid] += alpha * p[lsid, j, luid] + omega * s[lsid, j, luid]
                    r[lsid, j, luid] = rj
                    r2 += rj * rj
                    rhat2 += rj * rhat[lsid, j, luid]
                dot_tmp[lsid, luid] = r2        # ||r||^2
                dot_tmp2[lsid, luid] = rhat2    # (r @ rhat)    --> rho
            else:
                dot_tmp[lsid, luid] = 0.0
                dot_tmp2[lsid, luid] = 0.0
            cuda.syncthreads()

            if (gsid == 0) and (luid == 0):
                print("x(0,0)=", x[lsid, 0, 0], " x(0,1)=", x[lsid, 0, 1], " x(1,0)=", x[lsid, 1, 0], " x(1,1)=", x[lsid, 1, 1])
                print("r(0,0)=", r[lsid, 0, 0], " r(0,1)=", r[lsid, 0, 1], " r(1,0)=", r[lsid, 1, 0], " r(1,1)=", r[lsid, 1, 1])
            cuda.syncthreads()

            # 4/7) Check convergence and max iteratis
            if sys_leader and s_lin_terminated[lsid]==0:
                # Increase iteration counter
                s_num_iters[lsid] += 1
                # reduction --> rho
                rho = 0.0
                r2  = 0.0
                for i in range(UPS):
                    r2  += dot_tmp[lsid, i]
                    rho += dot_tmp2[lsid, i]
                if not math.isfinite(r2) or not math.isfinite(rho) or r2 < s_tol[lsid] * s_tol[lsid]:
                    # BiCGSTAB breakdown or convergence
                    s_lin_terminated[lsid] = 1
                    cuda.atomic.add(s_num_lin_terminated, 0, 1)
                elif (s_num_iters[lsid] >= MAXITER) and s_lin_terminated[lsid] == 0:
                    # Maximum iteration is reached terminate the system
                    # Avoid double terminat
                    s_lin_terminated[lsid] = 1
                    cuda.atomic.add(s_num_lin_terminated, 0, 1)
                else:
                    # System is not terminated --> calculete new beta
                    s_beta[lsid] = (rho / (s_rho[lsid] + eps2)) * (s_alpha[lsid] / (s_omega[lsid] + eps2))
                    s_rho[lsid] = rho
            cuda.syncthreads()

            # 4/8) mat-vec v = (A @ p)
            if sys_active and s_lin_terminated[lsid] == 0:
                per_block_implicit_mat_vec(UPS, gsid, luid, cpf, cpt, mx, p[lsid], Av[lsid])        # v = (A @ p)
            cuda.syncthreads()

            # 4/9 update p vecor
            # p = r + beta * (p - omega * v)
            if sys_active and s_lin_terminated[lsid] == 0:
                beta = s_beta[lsid]
                omega = s_omega[lsid]
                for j in range(CPS):
                    pj = p[lsid, j, luid]
                    p[lsid, j, luid] = r[lsid, j, luid] + beta * (pj - omega * Av[lsid, j, luid])
            cuda.syncthreads()
                    
            if (gsid == 0) and (luid == 0):
                print("v(0,0)=", Av[lsid, 0, 0], " v(0,1)=", Av[lsid, 0, 1], " v(1,0)=", Av[lsid, 1, 0], " v(1,1)=", Av[lsid, 1, 1])
                print("p(0,0)=", p[lsid, 0, 0], " p(0,1)=", p[lsid, 0, 1], " p(1,0)=", p[lsid, 1, 0], " p(1,1)=", p[lsid, 1, 1])
            cuda.syncthreads()
                 



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
        # --- Local alias ---
        active = (lsid < SPB) and (gsid < NS) and (s_flags[lsid, IS_TERMINATED] == 0)
        
        # --- Shared buffers ---
        s_relative_error = cuda.shared.array((SPB, ), dtype=nb.float64)     # per-system min tol/error
        s_rel_tmp        = cuda.shared.array((SPB, UPS), dtype=nb.float64)  # per-thread temporal
        s_upd_tmp        = cuda.shared.array((SPB, UPS), dtype=nb.int32)    # per-thread accept (AND)
        
        # --- per-system Init: leader set temp values --
        if active and luid == 0:
            s_relative_error[lsid]     = 1.0e30
            s_flags[lsid, UPDATE_STEP] = 1
            s_flags[lsid, IS_FINITE]   = 1
        cuda.syncthreads()
        
        # --- per-thread: local relative error calculation and accept flag --
        l_relative_error = 1e30
        l_update_step    = 1
        if active:
            for i in range(UD):
                error_tolerance  =  max(RTOL[i]*max(abs(l_next_state[i]), abs(l_actual_state[i])), ATOL[i])
                l_update_step   &=  (l_error[i] < error_tolerance)
                l_relative_error = min(l_relative_error, error_tolerance / l_error[i])
            if not math.isfinite(l_relative_error):
                s_flags[lsid, IS_FINITE] = 0
                l_relative_error         = 0
                print("Error Calculation: Global System ID: ", gsid, " State is not finite")
            s_rel_tmp[lsid, luid] = l_relative_error
            s_upd_tmp[lsid, luid] = l_update_step

            #print("l_relative_error", gsid, l_relative_error)

        cuda.syncthreads()

        # --- per-system reduction: leader only
        if active and luid == 0:
            s_rel = s_rel_tmp[lsid, 0]       # per-system relative error (MIN)
            s_upd = s_upd_tmp[lsid, 0]       # per-system update flag
            # --- units ---
            for i in range(1, UPS):
                # min relative error
                if s_rel_tmp[lsid, i] < s_rel:
                    s_rel = s_rel_tmp[lsid, i]
                # AND accept 
                s_upd &= s_upd_tmp[lsid, i]

            s_relative_error[lsid] = s_rel
            if s_upd == 0:
                s_flags[lsid, UPDATE_STEP] = 0

        cuda.syncthreads()

        # --- per-system: new time step, leader only
        if active and luid == 0:
            power = 0.2 if s_flags[lsid, UPDATE_STEP] == 1 else 0.5
            time_step_multiplicator = 0.9 * math.pow(s_relative_error[lsid], power)

            # -- Check finite time step --
            if not math.isfinite(time_step_multiplicator):
                s_flags[lsid, IS_FINITE] = 0
                s_flags[lsid, UPDATE_STEP] = 0
                time_step_multiplicator = time_step_shrink_limit
                print("TimeStep Calculation: Global System ID: ", gsid, " State is not finite")

            # -- Check min time step --
            if s_flags[lsid, IS_FINITE] == 0:
                # Infinite state --> Reduce the timestep
                if s_time_step[lsid] < time_step_min * 1.01:
                    s_flags[lsid, USER_TERMINATED] = 1
                    print("TimeStep Reduction: Global System ID: ", gsid," State is not a finite and the minimum step size is reached")
            else:
                # Finite state --> Check minimum step size is reached --> terminate the system to avoid crash
                if s_time_step[lsid] < time_step_min*1.01:
                    s_flags[lsid, UPDATE_STEP] = 0
                    s_flags[lsid, USER_TERMINATED] = 1
                    print("TimeStep Reduction: Global System ID: ", gsid," State is finite but the minimum step size is reached")

            # -- Finally, set the new time step value
            if s_flags[lsid, USER_TERMINATED] == 0:
                # - Clip the multiplicator 
                time_step_multiplicator = min(time_step_multiplicator, time_step_growth_limit)
                time_step_multiplicator = max(time_step_multiplicator, time_step_shrink_limit)

                # - Clip the time step -
                new_time_step = s_time_step[lsid] * time_step_multiplicator
                new_time_step = min(new_time_step, time_step_max)
                new_time_step = max(new_time_step, time_step_min)

                s_new_time_step[lsid] = new_time_step
            
        cuda.syncthreads()


    # --- DENSE OUTPUT ---
    @cuda.jit(nb.void(
            nb.int32,
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
            nb.int32[:,::]
    ), device=True, inline=True)
    def block_store_dense_output(
            gtid,
            gsid,
            lsid,
            luid,
            dense_output_time_instances,
            dense_output_states,
            l_actual_state,
            s_actual_time,
            s_dense_output_time,
            s_dense_output_time_index,
            l_dense_output_min_time_step,
            time_domain_end,
            s_flags):
        
        active = (lsid < SPB) and (gsid < NS) and (s_flags[lsid, IS_TERMINATED] == 0)
        s_store_flag = cuda.shared.array((SPB, ), dtype=nb.int32)
        s_store_idx  = cuda.shared.array((SPB, ), dtype=nb.int32)

        # -- Per-system -- Check Storage and time limit conditions (leader)
        if active and luid == 0:
            do_store = 0
            if (s_dense_output_time_index[lsid] < NDO) and \
                (s_dense_output_time[lsid] <= s_actual_time[lsid]):
                do_store = 1
                # -- Store time instances and indices --
                t_idx = s_dense_output_time_index[lsid]      # Last index (store here)
                dense_output_time_instances[t_idx * NS + gsid] = s_actual_time[lsid]

                # -- Prepare for the next sample --
                s_dense_output_time_index[lsid] = t_idx + 1  # New index 
                s_dense_output_time[lsid] = min(s_actual_time[lsid] + l_dense_output_min_time_step, time_domain_end)

                s_store_idx[lsid] = t_idx
            s_store_flag[lsid] = do_store

        cuda.syncthreads()

        # -- Per-thread -- Store dense states
        if active and s_store_flag[lsid] == 1:
            t_idx = s_store_idx[lsid]
            base_idx = t_idx * NUM_TOTAL_THREADS * UD + gtid
            for i in range(UD):
                dense_output_states[base_idx + i * NUM_TOTAL_THREADS] = l_actual_state[i]

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
            nb.float64[:,:,::],
            nb.float64[:,:,:,:],
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
        s_coupling_terms,
        s_coupling_matrices,
        l_unit_accessories,
        s_sytem_accessories,
        s_actual_time,
        s_time_step,
        s_flags):
        
        active = (lsid < SPB) and (gsid < NS) and (s_flags[lsid, IS_TERMINATED] == 0)
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
        if active:
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k1, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k1, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        #per_block_bicgstab(
        #    gsid, lsid, luid, 
        #    k1, l_coupling_factor, s_coupling_terms[lsid], s_coupling_matrices[lsid],
        #    s_flags)

        #cuda.syncthreads()

        # --- K2 ---
        if active:
            t = t0 + dt * (1.0 / 5.0)
            for i in range(UD):
                x[i] = l_actual_state[i] + dt * (1.0 / 5.0) * k1[i]
            
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k2, x, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k2, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        # --- K3 ---
        if active:
            t = t0 + dt * (3.0 / 10.0)
            for i in range(UD):
                x[i] = l_actual_state[i] \
                        + dt * (  (3.0 / 40.0) * k1[i] \
                                + (9.0 / 40.0) * k2[i] )
                
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k3, x, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k3, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        # --- K4 ---
        if active:
            t = t0 + dt * (3.0 / 5.0)
            for i in range(UD):
                x[i] = l_actual_state[i] \
                        + dt * (  (3.0 / 10.0) * k1[i] \
                                - (9.0 / 10.0) * k2[i] \
                                + (6.0 / 5.0)  * k3[i] )
            
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k4, x, l_coupling_factor, 
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k4, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        # --- K5 ---
        if active:
            t = t0 + dt
            for i in range(UD):
                x[i] = l_actual_state[i] \
                        + dt * (- (11.0 / 54.0) * k1[i] \
                                + (5.0 / 2.0 )  * k2[i] \
                                - (70.0 / 27.0) * k3[i] \
                                + (35.0 / 27.0) * k4[i] )
            
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k5, x, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k5, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        # --- K6 ---
        if active:
            t = t0 + dt * (7.0 / 8.0)
            for i in range(UD):
                x[i] = l_actual_state[i] \
                        + dt * ( (1631.0/55296.0) * k1[i] \
                                + (175.0/512.0)   * k2[i] \
                                + (575.0/13824.0) * k3[i] \
                                + (44275.0/110592.0) * k4[i] \
                                + (253.0/4096.0)  * k5[i] )
            
            per_block_ode_function(
                UPS, gsid, luid, t, 
                k6, x, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])
        
        cuda.syncthreads()

        if active:
            per_block_explicit_coupling(
                UPS, gsid, luid, t,
                k6, l_actual_state, l_coupling_factor,
                l_unit_parameters, s_global_parameters,
                s_system_parameters[lsid], s_dynamic_parameres[lsid],
                s_coupling_terms[lsid], s_coupling_matrices[lsid])

        cuda.syncthreads()

        # --- NEW STATE AND ERROR ---
        if active:
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

        s_coupling_terms = cuda.shared.array((SPB, NCT, UPS), dtype=nb.float64)

        s_coupling_matrices = cuda.shared.array((SPB, NC, UPS, UPS_P), dtype=nb.float64)   # Figure out the shape!

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
        # - TODO: Not optimal but works... improve later
        lid = cuda.threadIdx.x                   # Local System ID
        gid = lid + cuda.blockIdx.x * SPB        # Global System ID

        if (lid < SPB) and (gid < NS):
            # - Simulation Time -
            s_time_domain[lid, 0] = time_domain[gid]
            s_time_domain[lid, 1] = time_domain[gid + NS]
            s_actual_time[lid] = s_time_domain[lid, 0]
            s_time_step[lid] = s_new_time_step[lid] = time_step_init

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
            for mx_id in range(NC):
                for row in range(UPS):
                    offset = (((gid * NC + mx_id) *  UPS + row) *  UPS)
                    for col in range(UPS):
                        s_coupling_matrices[lid, mx_id, row, col] = coupling_matrices[offset + col]

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
        
        if (lsid < SPB) and (gsid < NS) and (s_flags[lsid][IS_TERMINATED] == 0):
            # - INITIALIZATION -
            l_dense_output_min_time_step = (s_time_domain[lsid, 1] - s_time_domain[lsid, 0]) / (NDO - 1)
            per_block_initialization(
                UPS,
                gsid,
                luid,
                s_actual_time[lsid],
                s_time_domain[lsid],
                l_actual_state,
                l_coupling_factors,
                l_unit_parameters,
                s_global_parameters,
                s_system_parameters[lsid],
                s_dynamic_parameters[lsid],
                s_coupling_terms[lsid],
                s_coupling_matrices[lsid])
        cuda.syncthreads()
        
        if (NE > 0):
                # per_block_event_function
                # per_block_event_time_step_control
                pass
        cuda.syncthreads()

        if (NDO > 0): 
            block_store_dense_output(
                gtid,
                gsid,
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
                s_flags
            )
            print(l_dense_output_min_time_step)
        cuda.syncthreads()


        # - SOLVER MAIN LOOP -
        while (s_terminated_systems_per_block[0] < SPB):
            # 1) TIME STEP INIT per-system init (leader)
            if (lsid < SPB) and (s_flags[lsid, IS_TERMINATED] == 0) and (luid == 0):
            #if ((lid < SPB) and s_flags[lid, IS_TERMINATED] == 0):
                s_flags[lsid, UPDATE_STEP] = 1
                s_flags[lsid, IS_FINITE]   = 1

                remain = s_time_domain[lsid, 1] - s_actual_time[lsid]
                if (s_new_time_step[lsid] > remain):
                    s_time_step[lsid] = remain
                    s_flags[lsid, END_TIME_DOMAIN] = 1          # Last Step!
                else:
                    s_time_step[lsid] = s_new_time_step[lsid]
                    s_flags[lsid, END_TIME_DOMAIN] = 0

            cuda.syncthreads()

            # 2) FORWARD STEPPING & EROOR CONTROL
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
                    s_coupling_terms,
                    s_coupling_matrices,
                    l_unit_accessories,
                    s_system_accessories,
                    s_actual_time,
                    s_time_step,
                    s_flags)
                
                cuda.syncthreads()

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

                cuda.syncthreads()

            # 3) NEW EVENT VALUE AND TIME STEP CONTROL -
            if NE > 0:
                # per block event function
                # per block event time_step_control
                pass


            # 4) SUCCESFULL TIME STEPPING UPDATE TIME AND STATE, per-system (leader)
            if (lsid < SPB) and (gsid < NS) and (s_flags[lsid, IS_TERMINATED] == 0) and \
                (s_flags[lsid, UPDATE_STEP] == 1):
                if luid == 0:
                    s_actual_time[lsid] += s_time_step[lsid]
                    #if gtid == 0:
                    #print("s_actual_time", gsid, s_actual_time[lsid])

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
                    gsid,
                    lsid,
                    luid,
                    dense_output_time_instances,
                    dense_output_states,
                    l_actual_state,
                    s_actual_time,
                    s_dense_output_time,
                    s_dense_output_time_index,
                    l_dense_output_min_time_step,
                    s_time_domain[lsid][1],
                    s_flags,
                )

            cuda.syncthreads()

            # 5) CHECK TERMINATION per system (leader)
            if (lsid < SPB) and (gsid < NS) and (luid == 0):
                done = 0
                if (s_flags[lsid, USER_TERMINATED] == 1) or (s_flags[lsid, EVENT_TERMINATED] == 1):
                    done = 1    # Event or user terminated TODO: add global flag
                elif (s_flags[lsid, UPDATE_STEP] == 1) and (s_flags[lsid, END_TIME_DOMAIN] == 1):
                    done = 1    # Last timestep is accepted and the end of the integration domain is reached

                if done and (s_flags[lsid, IS_TERMINATED] == 0):
                    # New terminal event!
                    s_flags[lsid, IS_TERMINATED] = 1
                    cuda.atomic.add(s_terminated_systems_per_block, 0, 1)
                    print(s_terminated_systems_per_block[0])                          

            cuda.syncthreads()
                
            # Avoid endless loop during implementation --> increas terminated system coutn after every iteration
            #if cuda.threadIdx.x == 0:
            #    cuda.atomic.add(s_terminated_systems_per_block, 0, 1)

        # - FINALIZATION -
        if (lsid < SPB) and (gsid < NS):
            per_block_finalization(
                UPS,
                gsid,
                luid,
                s_actual_time[lsid],
                s_time_domain[lsid],
                l_actual_state,
                l_coupling_factors,
                l_unit_parameters,
                s_global_parameters,
                s_system_parameters[lsid],
                s_dynamic_parameters[lsid],
                s_coupling_terms[lsid],
                s_coupling_matrices[lsid]
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
            #print(gtid, s_coupling_matrices[1, 0, 0, 0])
            #print(gtid, s_coupling_matrices[1, 0, 0, 1])
            #print(gtid, s_coupling_matrices[1, 0, 1, 0])
            #print(gtid, s_coupling_matrices[1, 0, 1, 1])

            #print(gtid, s_coupling_matrices[1, 1, 0, 0])
            #print(gtid, s_coupling_matrices[1, 1, 0, 1])
            #print(gtid, s_coupling_matrices[1, 1, 1, 0])
            #print(gtid, s_coupling_matrices[1, 1, 1, 1])

            #print(gtid, s_coupling_matrices[1, 2, 0, 0])
            #print(gtid, s_coupling_matrices[1, 2, 0, 1])
            #print(gtid, s_coupling_matrices[1, 2, 1, 0])
            #print(gtid, s_coupling_matrices[1, 2, 1, 1])








    return coupled_system_per_block

