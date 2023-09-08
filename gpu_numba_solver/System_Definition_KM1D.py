import numba as nb
from numba import cuda
import math


# ---------- Properties of the acoustic field is --------------
@cuda.jit(nb.float64(nb.float64, nb.float64, nb.float64[:]),
          device=True, inline=True)
def _PA(t, x, cp):
    return cp[9]  * math.sin(2*math.pi*x*cp[26]*cp[21]) * math.sin(2*math.pi*t*cp[11]*cp[25]) \
         + cp[10] * math.sin(2*math.pi*x*cp[26]*cp[22]) * math.sin(2*math.pi*t*cp[12]*cp[25] + cp[13])


@cuda.jit(nb.float64(nb.float64, nb.float64, nb.float64[:]),
          device=True, inline=True)
def _PAT(t, x, cp):
    return  cp[9]  * cp[11] * math.sin(2*math.pi*x*cp[26]*cp[21]) * math.cos(2*math.pi*t*cp[11]*cp[25]) \
          + cp[10] * cp[12] * math.sin(2*math.pi*x*cp[26]*cp[22]) * math.cos(2*math.pi*t*cp[12]*cp[25] + cp[13])


@cuda.jit(nb.float64(nb.float64, nb.float64, nb.float64[:]),
          device=True, inline=True)
def _GRADP(t, x, cp):
    return cp[9]  * cp[19] * math.cos(2*math.pi*x*cp[26]*cp[21]) * math.sin(2*math.pi*t*cp[11]*cp[25]) \
         + cp[10] * cp[20] * math.cos(2*math.pi*x*cp[26]*cp[22]) * math.sin(2*math.pi*t*cp[12]*cp[25] + cp[13] )


@cuda.jit(nb.float64(nb.float64, nb.float64, nb.float64[:]),
          device=True, inline=True)
def _UAC(t, x, cp):
    return  cp[9]  * cp[23] * math.cos(2*math.pi*x*cp[26]*cp[21]) * math.cos(2*math.pi*t*cp[11]*cp[25]) \
          + cp[10] * cp[23] * math.cos(2*math.pi*x*cp[26]*cp[22]) * math.cos(2*math.pi*t*cp[12]*cp[25] + cp[13] )


# ----------- ODE FUNCTION ---------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_ode_function(tid, t, dx, x, acc, cp):

    ''' TODO: Add correct docstring
    Implement thr RHS of the ODE here
    dx[:] = f[:](t, x[:], cp[:])

    '''

    # Keller--Miksis equation
    rx1 = 1.0 / x[0]
    p = rx1**cp[8]

    N = (cp[0] + cp[1]*x[2]) * p \
    - cp[2] * (1 + cp[7]*x[2]) - cp[3]* rx1 - cp[4]*x[2]*rx1 \
    - 1.5 * (1.0 - cp[7]*x[2] * (1.0/3.0))*x[2]*x[2] \
    - (1 + cp[7]*x[2]) * cp[5] * _PA(t, x[1], cp) - cp[6] * _PAT(t, x[1], cp) * x[0] \
    + 0.25 * x[3]*x[3]*cp[15]  

    D = x[0] - cp[7]*x[0]*x[2] + cp[4]*cp[7]

    rD = 1.0 / D

    # Translation Motion
    Fb1 = - cp[17]*x[0]*x[0]*x[0] * _GRADP(t, x[1], cp)            # Primary Bjerknes Force
    Fd  = - cp[18]*x[0] * (x[3]*cp[24] - _UAC(t, x[1], cp) )       # Drag Force

    # Composition of the ode functions
    dx[0] = x[2]
    dx[1] = x[3]
    dx[2] = N*rD
    dx[3] = 3*(Fb1+Fd)*cp[16]*rx1*rx1*rx1 - 3.0*x[2]*rx1*x[3] 

    # Integrate the Bjerknes force to get the time-average
    dx[4] = Fb1
    dx[5] = Fd
    


# ---------- ACCESSORIES --------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_timesteps(tid, t, x, acc, cp):
    pass


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_initialization(tid, t, td, x, acc, cp):

    x[4] = 0.0         # Primary Bjerknes Force Integral
    x[5] = 0.0         # Drag Forca Integral
    acc[2] = x[1]      # Initial Bubble Position


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp):

    # Calculate the time-averaged Bjerknes force
    dt = td[1] - td[0]
    acc[0] = x[4] / dt              # Time-Averaged Primary Bjerknes force
    acc[1] = x[5] / dt              # Time-Averaged Drag Force
    acc[3] = x[1]                   # Last Bubble Position
    acc[4] = (acc[3]-acc[2]) / dt   # Average translational velocity 

    # Increas the time domain by dt
    td[0] += 1.0 * dt
    td[1] += 1.0 * dt

