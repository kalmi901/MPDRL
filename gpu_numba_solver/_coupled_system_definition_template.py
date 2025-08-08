import numba as nb
from numba import cuda
import math

# ---------- FUNCTION SIGNATURES --------
__ODE_FUN_SIGNATURE = nb.void(nb.int32, nb.int32, nb.float64, 
                              nb.float64[:], nb.float64[:], nb.float64[:],
                              nb.float64[:], nb.float64[:], 
                              nb.float64[:], nb.float64[:])

# ----------- ODE FUNCTION -------------
@cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_ode_function(gsid, luid, t, dx, x, cpf, up, gp, sp, dp):

    """
    TODO
    Implement
    A[:, :] dx[:] = f[:](t, x[:], up[:]) 
    """

    pass


cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_initialization(gsid, luid, t, td, x, cpf, up, gp, sp, dp):
    pass


cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_finalization(gsid, luid, t, td, x, cpf, up, gp, sp, dp):
    pass

