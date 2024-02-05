import numba as nb
from numba import cuda
import math

# ----------------- ODE Solver Parameters ------------------
DEFAULT_SOLVER_OPTS = {
    "SD"  : 2,              # System Dimension (r, u)
    "NCP" : 6,              # Number of Control Parameters
    "NACC": 2,              # Number of Accessories
    "NE"  : 1,              # Number of Events
    "SOLVER": "RKCK45",     # ODE-solver algo
    "BLOCKSIZE": 64,        # Number of Threads per block
    "ATOL" : 1e-9,          # Absolute Tolerance
    "RTOL" : 1e-9,          # Relative Tolerance
    "NT": 1,                # Number of threads (at least on active thread requidred)
    "ETOL": 1e-9,           # Event Tolerance
    "EDIR":-1,              # Event Direction
}

# -------------- Global Model Constants Parameters ----------
# Material Properties
DEFAULT_MAT_PROPS = {
    "PV"  : 3.166775638952003e+03,    # Vapour Pressure [Pa]
    "RHO" : 9.970639504998557e+02,    # Liquid Density  [kg/m**3]  
    "ST"  : 0.071977583160056,        # Surface Tension [N/m]
    "VIS" : 8.902125058209557e-04,    # Liquid Viscosity [Pa s]
    "CL"  : 1.497251785455527e+03,    # Liquid Sound Speed [m/s]
    "P0"  : 1.0e5,                    # Ambient Pressure [Pa]
    "PE"  : 1.4,                      # Polytrophic Exponent
}

# Equation Properties (Default values in SI Units
DEFAULT_EQ_PROPS = {
    "R0"    : 110.0* 1e-6,    # Equilibrium Radius [m]
    "FREQ"  : 25.0 * 1e3,     # Excitation frequencies (kHz)
    "PA"    : 0.2  * 1e5,     # Pressure Amplitude (bar)
}

# ----------------- Calculation of control parameters -----------------------
CP = {
    0 : lambda **kwargs : (kwargs["PV"] - kwargs["P0"]) / kwargs["RHO"] * (1.0 / kwargs["FREQ"] / kwargs["R0"])**2,
    1 : lambda **kwargs : (2.0 * kwargs["ST"] / kwargs["R0"] + kwargs["P0"] - kwargs["PV"]) / kwargs["RHO"] * (1.0 / kwargs["FREQ"] / kwargs["R0"])**2,
    2 : lambda **kwargs : 3.0 * kwargs["PE"],
    3 : lambda **kwargs : 2.0 * kwargs["ST"] / kwargs["RHO"] / kwargs["R0"] * (1.0 / kwargs["FREQ"] / kwargs["R0"])**2,
    4 : lambda **kwargs : 4.0 * kwargs["VIS"] / kwargs["RHO"] / (kwargs["R0"]**2) * (1.0 / kwargs["FREQ"]),
    5 : lambda **kwargs : kwargs["PA"] / kwargs["RHO"] * (1.0 / kwargs["FREQ"] / kwargs["R0"])**2
}


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_ode_function(tid, t, dx, x, acc, cp):

    ''' TODO: Add correct docstring
    Implement thr RHS of the ODE here
    dx[:] = f[:](t, x[:], cp[:])

    '''
    rx0 = 1.0 / x[0]

    dx[0] = x[1]
    dx[1] = cp[0] + cp[1] * rx0**cp[2] - cp[3] * rx0 - cp[4] * x[1] * rx0 - cp[5] * math.sin(2*math.pi*t) - 1.5 * x[1] * x[1] * rx0
    


# ---------- ACCESSORIES --------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_timesteps(tid, t, x, acc, cp):
    acc[0] = max(acc[0], x[0])


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_initialization(tid, t, td, x, acc, cp):
    acc[0] = x[0]


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp):
    # Increase the time domain
    #td[0] += 1.0
    td[1] += 1.0


# -------------- EVENTS ---------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_event_function(tid, t, ev, x, acc, cp):
    ev[0] = x[1]

@cuda.jit(nb.boolean(nb.int32, nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_event_detection(tid, idx, t, td, x, acc, cp):
    acc[1] = t
    td[0] = t
    
    return True