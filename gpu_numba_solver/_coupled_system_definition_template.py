import numba as nb
from numba import cuda
import math

# ---------- FUNCTION SIGNATURES --------
__ODE_FUN_SIGNATURE = nb.void(nb.int32, nb.int32, nb.int32, nb.float64, 
                              nb.float64[:], nb.float64[:], nb.float64[:],
                              nb.float64[:], nb.float64[:], 
                              nb.float64[:], nb.float64[:],
                              nb.float64[:,:], nb.float64[:,:,:])

__MAT_VEC_SIGNATURE = nb.void(nb.int32, nb.int32, nb.int32,
                              nb.float64[:], nb.float64[:,:], nb.float64[:,:,:],
                              nb.float64[:,:], nb.float64[:,:])

# ----------- ODE FUNCTION -------------
@cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_ode_function(ups, gsid, luid, t, dx, x, cpf, up, gp, sp, dp, cpt, mx):

    """
    TODO
    Implement
    A[:, :] dx[:] = f[:](t, x[:], up[:]) 
    """

    pass

@cuda.jit(__ODE_FUN_SIGNATURE)
def per_block_explicit_coupling(ups, gsid, luid, t, dx, x, cpf, up, gp, sp, dp, cpt, mx):
    pass

cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_initialization(ups, gsid, luid, t, td, x, cpf, up, gp, sp, dp, cpt, mx):
    pass


cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
def per_block_finalization(ups, gsid, luid, t, td, x, cpf, up, gp, sp, dp, cpt, mx):
    pass


@cuda.jit(__MAT_VEC_SIGNATURE, device=True, inline=True)
def per_block_implicit_mat_vec(ups, gsid, luid, cpf, cpt, mx, v, Av):
    pass
