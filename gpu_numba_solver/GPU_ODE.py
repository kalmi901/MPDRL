import numba as nb
from numba import cuda
import numpy as np
import math
from typing import Union
from types import ModuleType

try:
    from gpu_numba_solver.System_Definition_Template import *
except:
    from System_Definition_Template import *


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
__SYSTEM_DEF_FUNCTIONS = ["per_thread_ode_function", "per_thread_action_after_timesteps", "per_thread_initialization", "per_thread_finalization"]

def setup(system_definition: ModuleType):
    for func_name in __SYSTEM_DEF_FUNCTIONS:
        if hasattr(system_definition, func_name):
            globals()[func_name] = getattr(system_definition, func_name)



class SolverObject():

    def __init__(self, 
                number_of_threads : int,
                system_dimension  : int,
                number_of_control_parameters : int,
                number_of_accessories : int = 1,
                threads_per_block : int = 64,
                method : str = "RKCK45",
                abs_tol : Union[list, float] = None,
                rel_tol : Union[list, float] = None,
                min_step: float = 1e-16,
                max_step: float = 1.0e6,
                init_step:float = 1e-2,
                growth_limit:float = 5.0,
                shrink_limit:float = 0.1,
                ):
        

        # ----- Constant (private) properties -------------
        self._system_dimension = system_dimension
        self._number_of_threads = number_of_threads
        self._number_of_control_parameters = number_of_control_parameters
        self._number_of_accessories = number_of_accessories
        self._threads_per_block = threads_per_block
        self._blocks_per_grid = self._number_of_threads // self._threads_per_block + (0 if self._number_of_threads % self._threads_per_block == 0 else 1 )
        self._method = method


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
            self._rel_tol = np.array(abs_tol, dtype=np.float64)
        else:
            self._rel_tol = np.full((system_dimension, ), abs_tol, dtype=np.float64)

        assert len(self._abs_tol) == system_dimension
        assert len(self._rel_tol) == system_dimension

        # ----- Public properties with default values ----
        self.time_step_init         = init_step
        self.time_step_max          = max_step
        self.time_step_min          = min_step
        self.time_step_growth_limit = growth_limit
        self.time_step_shrink_limit = shrink_limit


        # ----- Constant (private) array sizes -----
        self._size_of_time_domain  = number_of_threads * 2
        self._size_of_actual_time  = number_of_threads
        self._size_of_actual_state = system_dimension * number_of_threads
        self._size_of_accessories  = number_of_accessories * number_of_threads
        self._size_of_control_parameters = number_of_control_parameters * number_of_threads
        # TODO: Add more if necessary

        # ---- Host side (private) arrays -----
        self._h_time_domain  = cuda.pinned_array((self._size_of_time_domain, ), dtype=np.float64)
        self._h_actual_time  = cuda.pinned_array((self._size_of_actual_time, ), dtype=np.float64)
        self._h_actual_state = cuda.pinned_array((self._size_of_actual_state, ), dtype=np.float64)
        self._h_accessories  = cuda.pinned_array((self._size_of_accessories, ), dtype=np.float64)
        self._h_control_parameter = cuda.pinned_array((self._size_of_control_parameters, ), np.float64)

        # ---- Device side (private) arrays -----
        self._d_time_domain  = cuda.device_array_like(self._h_time_domain)
        self._d_actual_time  = cuda.device_array_like(self._h_actual_time)
        self._d_actual_state = cuda.device_array_like(self._h_actual_state)
        self._d_accessories  = cuda.device_array_like(self._h_accessories)
        self._d_control_parameter = cuda.device_array_like(self._h_control_parameter)

        # ---- Create / Compile the Kenel function -----
        self._myfckin_kernel = _RKCK45_kernel(  self._number_of_threads,
                                                self._system_dimension,
                                                self._number_of_control_parameters,
                                                self._number_of_accessories,
                                                self._method,
                                                self._abs_tol,
                                                self._rel_tol)
        

    # ---- Interface to call the GPU Kernel function ------
    def solve_my_ivp(self) :
        self._myfckin_kernel[self._blocks_per_grid, self._threads_per_block](
                                self._number_of_threads,
                                self._d_actual_state,
                                self._d_control_parameter,
                                self._d_accessories,
                                self._d_time_domain,
                                self._d_actual_time,
                                self.time_step_init,
                                self.time_step_max,
                                self.time_step_min,
                                self.time_step_growth_limit,
                                self.time_step_shrink_limit)
        cuda.synchronize()


    # ----  Helper functions -----
    #-----------------------------
    def set_host(self, thread_id: int, property: str, index: int, value: float) :

        idx = thread_id + index * self._number_of_threads
        
        if property == "time_domain":
            self._h_time_domain[idx] = np.float64(value)
        elif property == "actual_state":
            self._h_actual_state[idx] = np.float64(value)
        elif property == "control_parameter":
            self._h_control_parameter[idx] = np.float64(value)
        elif property == "accessories":
            self._h_accessories[idx] = np.float64(value)
        else:
            print("Nemfasza: SetHost")

    def get_host(self, thread_id: int, property: str, index: int):

        idx = thread_id + index * self._number_of_threads

        if property == "time_domain":
            return self._h_time_domain[idx]
        elif property == "actual_state":
            return self._h_actual_state[idx]
        elif property == "control_parameter":
            return self._h_control_parameter[idx]
        elif property == "accessories":
            return self._h_accessories[idx]
        else:
            print("Nemfasza: GetHost")
            return None

    def syncronize_h2d(self, property:str):
        if property == "time_domain":
            self._d_time_domain.copy_to_device(self._h_time_domain)
        elif property == "control_parameter":
            self._d_control_parameter.copy_to_device(self._h_control_parameter)
        elif property == "actual_state":
            self._d_actual_state.copy_to_device(self._h_actual_state)
        elif property == "accessories":
            self._d_accessories.copy_to_device(self._h_accessories)
        elif property == "all":
            self._d_time_domain.copy_to_device(self._h_time_domain)
            self._d_control_parameter.copy_to_device(self._h_control_parameter)
            self._d_actual_state.copy_to_device(self._h_actual_state)
            self._d_accessories.copy_to_device(self._h_accessories)
        else:
            print("Nemfasza...")

        cuda.synchronize()

    def syncronize_d2h(self, property:str):
        if property == "time_domain":
            self._d_time_domain.copy_to_host(self._h_time_domain)
        elif property == "control_parameter":
            self._d_control_parameter.copy_to_host(self._h_control_parameter)
        elif property == "actual_state":
            self._d_actual_state.copy_to_host(self._h_actual_state)
        elif property == "accessories":
            self._d_accessories.copy_to_host(self._h_accessories)
        elif property == "all":
            self._d_time_domain.copy_to_host(self._h_time_domain)
            self._d_control_parameter.copy_to_host(self._h_control_parameter)
            self._d_actual_state.copy_to_host(self._h_actual_state)
            self._d_accessories.copy_to_host(self._h_accessories)
        else:
            print("Nemfasza...")

        cuda.synchronize()


# ----------  ALGO ---------------
# --------------------------------
# ----- Main Kernel Function -----
def _RKCK45_kernel( number_of_threads: int,
                    system_dimension: int,
                    number_of_control_parameters: int,
                    number_of_accessories: int,
                    method : str,
                    abs_tol : np.ndarray,
                    rel_tol : np.ndarray):
    
    # CONSTANT PARAMETERS ---------------
    # Create alieses
    NT  = number_of_threads
    SD  = system_dimension
    NCP = number_of_control_parameters
    NA  = number_of_accessories
    ALGO = method 
    ATOL = abs_tol
    RTOL = rel_tol

    # DEVICE FUNCTIONS CALLED IN THE KERNEL
    # -------------------------------------

    # Error Control -----------------------
    @cuda.jit(nb.void(
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
            print("State is not finite")
            l_is_finite = False

        # TODO: !!!!
        if l_is_finite == False:
            time_step_multiplicator = time_step_shrink_limit
            l_update_step = False

            if l_time_step < time_step_min*1.01:
                l_terminate_thread = True
                print('State is not a finite number even with the minimal step size')

        else:
            if (l_time_step < time_step_min*1.01):
                print('Minimum time step is reached')
                l_update_step = True

        time_step_multiplicator = min(time_step_multiplicator, time_step_growth_limit)
        time_step_multiplicator = max(time_step_multiplicator, time_step_shrink_limit)

        l_new_time_step = l_time_step * time_step_multiplicator
        l_new_time_step = min(l_new_time_step, time_step_max)
        l_new_time_step = max(l_new_time_step, time_step_min)

        return l_update_step, l_terminate_thread, l_new_time_step


    # STEPPER FUNCTIONS -----------------------------

    @cuda.jit(nb.boolean(
                nb.int32,
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
                                l_control_parameter)
        
        # K2 ----------------------------
        t = l_actual_time + dTp2

        for i in nb.prange(SD):
            x[i] = l_actual_state[i] + l_next_state[i] * dTp2

        per_thread_ode_function(tid,
                                t,
                                k1,
                                x, 
                                l_accessories,
                                l_control_parameter)
        
        # K3 ----------------------------
        for i in nb.prange(SD):
            l_next_state[i] = l_next_state[i] + 2.0 * k1[i]
            x[i] = l_actual_state[i] + k1[i] * dTp2

        per_thread_ode_function(tid,
                                t,
                                k1,
                                x,
                                l_accessories,
                                l_control_parameter)

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
                                l_control_parameter)

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
                nb.float64,
                nb.float64
    ), device=True, inline=True)
    def per_thread_stepper_RKCK45(tid,
                           l_next_state,
                           l_actual_state,
                           l_error,
                           l_control_parameter,
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
                                l_control_parameter)


        # K2 ---------------------
        t = l_actual_time + l_time_step * (1.0 / 5.0)
        for i in nb.prange(SD):
            x[i] = l_actual_state[i] + l_time_step * (1.0 / 5.0) * k1[i]
        
        per_thread_ode_function(tid,
                                t,
                                k2,
                                x, 
                                l_accessories,
                                l_control_parameter)
        
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
                                l_control_parameter)
        
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
                                l_control_parameter)
        
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
                                l_control_parameter)

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
                                l_control_parameter)
        
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
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                nb.float64,
                ),
                **CUDA_JIT_PROPERTIES)
    def single_system_per_thread(number_of_threads,
                                actual_state,
                                control_parameter,
                                accessories,
                                time_domain,
                                actual_time,
                                time_step_init,
                                time_step_max,
                                time_step_min,
                                time_step_growth_limit,
                                time_step_shrink_limit):
        
        tid = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x

        # RKCK 45 algo
        if tid < number_of_threads:

            # THREAD LOCAL MEMORY
            l_time_domain = cuda.local.array((2, ), dtype=nb.float64)
            l_actual_state = cuda.local.array((SD, ), dtype=nb.float64)
            l_next_state = cuda.local.array((SD, ), dtype=nb.float64)
            l_error = cuda.local.array((SD, ), dtype=nb.float64)
            l_control_parameter = cuda.local.array((NCP, ), dtype=nb.float64)
            l_accessories = cuda.local.array((NA, ), dtype=nb.float64)

            for i in nb.prange(2):
                l_time_domain[i] = time_domain[tid + i*NT]
            
            for i in nb.prange(SD):
                l_actual_state[i] = actual_state[tid + i*NT]

            for i in nb.prange(NCP):
                l_control_parameter[i] = control_parameter[tid + i*NT]

            for i in nb.prange(NA):
                l_accessories[i] = accessories[tid + i*NT]
            
            l_actual_time = l_time_domain[0]
            l_time_step = time_step_init
            l_new_time_step = time_step_init
            l_terminate = False      
    
            # INITIALIZATION
            per_thread_initialization(
                            tid,
                            l_actual_time,
                            l_time_domain,
                            l_actual_state,
                            l_accessories,
                            l_control_parameter)
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
                                    l_accessories,
                                    l_actual_time,
                                    l_time_step)
                
                    if l_is_finite == False:
                        print("State is not finite")
                        l_terminate = 1
                        l_new_time_step = time_step_init

                elif ALGO == "RKCK45":
                    l_is_finite = per_thread_stepper_RKCK45(
                                    tid,
                                    l_next_state,
                                    l_actual_state,
                                    l_error,
                                    l_control_parameter,
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


                # SUCCESFULL TIMESTEPPING UPDATE TIME AND STATE
                if l_update_step == True:
                    l_actual_time += l_time_step

                    for i in nb.prange(SD):
                        l_actual_state[i] = l_next_state[i]

                    # ACTION AFTER SUCCESFULL TIMESTEP
                    per_thread_action_after_timesteps(
                                        tid,
                                        l_actual_time,
                                        l_actual_state,
                                        l_accessories,
                                        l_control_parameter)

                    if l_time_domain_ends:
                        l_terminate = True

            # FINALIZATION ------------------------------------
            per_thread_finalization(
                            tid,
                            l_actual_time,
                            l_time_domain,
                            l_actual_state,
                            l_accessories,
                            l_control_parameter)

            cuda.syncthreads()
            # Integration ends, wrtite data back to global memory
            for i in nb.prange(2):
                time_domain[tid + i*NT] = l_time_domain[i]

            for i in nb.prange(SD):
                actual_state[tid + i*NT] = l_actual_state[i]

            for i in nb.prange(NCP):
                control_parameter[tid + i*NT] = l_control_parameter[i]

            for i in nb.prange(NA):
                accessories[tid + i*NT] = l_accessories[i]

            actual_time[tid] = l_actual_time


    return single_system_per_thread


# ----------------------------------
