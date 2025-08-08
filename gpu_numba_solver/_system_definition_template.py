import numba as nb
from numba import cuda
import math


# ----------- ODE FUNCTION ---------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_ode_function(tid, t, dx, x, acc, cp, dp, sp):

    ''' TODO: Add correct docstring
    Implement thr RHS of the ODE here
    dx[:] = f[:](t, x[:], cp[:])

    '''
    pass
    


# ---------- ACCESSORIES --------------
@cuda.jit(nb.boolean(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_timesteps(tid, t, x, acc, cp, dp, sp):
    return False


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_initialization(tid, t, td, x, acc, cp, dp, sp):
    pass


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp, dp, sp):
    pass

# -------------- EVENTS ---------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_event_function(tid, t, ev, x, acc, cp, dp, sp):
    pass

@cuda.jit(nb.boolean(nb.int32, nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_event_detection(tid, idx, t, td, x, acc, cp, dp, sp):
    return False
