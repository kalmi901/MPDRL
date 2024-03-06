import numba as nb
from numba import cuda
import math


# ----------------- ODE Solver Parameters ------------------
DEFAULT_SOLVER_OPTS = {
    "SD"  : 6,              # System Dimensios (r, u, x, v, Fb1, Fd)
    "NCP" : 27,             # Number of Control Parameters
    "NACC": 5,              # Number of Accessories
    "SOLVER": "RKCK45",     # ODE-solver algo
    "BLOCKSIZE": 64,        # Number of Threads per block
    "ATOL" : 1e-9,          # Absolute Tolerance
    "RTOL" : 1e-9,          # Relative Tolerance
    "NT": 1,                 # Number of threads (at least on active thread requidred)
}


# -------------- Global Model Constants Parameters ----------
# Material Properties
DEFAULT_MAT_PROPS = {
    "PV"  : 0.0,    # Vapour Pressure [Pa]
    "RHO" : 998.0,  # Liquid Density  [kg/m**3]  
    "ST"  : 0.0725, # Surface Tension [N/m]
    "VIS" : 0.001,  # Liquid Viscosity [Pa s]
    "CL"  : 1500,   # Liquid Sound Speed [m/s]
    "P0"  : 1.0e5,  # Ambient Pressure [Pa]
    "PE"  : 1.4,    # Polytrophic Exponent
}

# Equation Properties (Default values in SI Units
DEFAULT_EQ_PROPS = {
    "R0"    : 110 * 1e-6,     # Equilibrium Radius [m]
    "FREQ"  : [25.0 * 1e3,
               50.0 * 1e3],   # Excitation frequencies (kHz)
    "PA"    : [0.8 * 1e5,
               0.0 * 1e5],    # Pressure Amplitude (bar)
    "PS"    : 0.0,            # Phase Shift (radians)
    "REL_FREQ" : 25.0 * 1e3,  # Relative Frequency (kHz)
}


# ------------- Calculation of control parameters -----------
CP = {
    0 : lambda **kwargs : (2.0 * kwargs["ST"] / kwargs["R0"] + kwargs["P0"] - kwargs["PV"]) 
                          * (1.0 / kwargs["R0"] / kwargs["REL_FREQ"])**2.0 / kwargs["RHO"],
    1 : lambda **kwargs : (1.0 - 3.0*kwargs["PE"]) * (2 * kwargs["ST"] / kwargs["R0"] + kwargs["P0"]- kwargs["PV"]) 
                          * (1.0 / kwargs["R0"] / kwargs["REL_FREQ"]) / kwargs["CL"] / kwargs["RHO"],
    2 : lambda **kwargs : (kwargs["P0"] - kwargs["PV"]) * (1.0 / kwargs["R0"] / kwargs["REL_FREQ"])**2.0 / kwargs["RHO"], 
    3 : lambda **kwargs : (2.0 * kwargs["ST"] / kwargs["R0"] / kwargs["RHO"]) * (1.0 / kwargs["R0"] / kwargs["REL_FREQ"])**2.0,
    4 : lambda **kwargs : 4.0 * kwargs["VIS"] / kwargs["RHO"] / (kwargs["R0"]**2.0) * (1.0 / kwargs["REL_FREQ"]),
    5 : lambda **kwargs : ((1.0 / kwargs["R0"] / kwargs["REL_FREQ"])**2.0) / kwargs["RHO"],
    6 : lambda **kwargs : ((1.0 / kwargs["REL_FREQ"])** 2.0) / kwargs["CL"] / kwargs["RHO"] / kwargs["R0"],
    7 : lambda **kwargs : kwargs["R0"] * kwargs["REL_FREQ"] / kwargs["CL"],
    8 : lambda **kwargs : 3.0 * kwargs["PE"],
    # Physical Parameters
    9 : lambda **kwargs : kwargs["PA"][0],
    10: lambda **kwargs : kwargs["PA"][1],
    11: lambda **kwargs : 2.0 * math.pi * kwargs["FREQ"][0],
    12: lambda **kwargs : 2.0 * math.pi * kwargs["FREQ"][1],
    13: lambda **kwargs : kwargs["PS"],
    14: lambda **kwargs : kwargs["R0"],
    # Parameters for translation
    15: lambda **kwargs : (kwargs["CL"] / kwargs["REL_FREQ"] / kwargs["R0"])**2,
    16: lambda **kwargs : 1.0 / (kwargs["RHO"] * kwargs["CL"] * kwargs["REL_FREQ"] * (2.0 * math.pi) * (kwargs["R0"]**3)),
    17: lambda **kwargs : 4.0 * math.pi / 3.0 * kwargs["R0"]**3.0,
    18: lambda **kwargs : 12.0 * math.pi * kwargs["VIS"] * kwargs["R0"],
    # Acoustic Field Properties
    19: lambda **kwargs : 2.0 * math.pi / (kwargs["CL"] / kwargs["FREQ"][0]),
    20: lambda **kwargs : 2.0 * math.pi / (kwargs["CL"] / kwargs["FREQ"][1]),
    21: lambda **kwargs : kwargs["FREQ"][0] / kwargs["CL"],
    22: lambda **kwargs : kwargs["FREQ"][1] / kwargs["CL"],
    23: lambda **kwargs : 1.0 / kwargs["RHO"] / kwargs["CL"],
    24: lambda **kwargs : kwargs["CL"],
    25: lambda **kwargs : 1.0 / (2.0 * math.pi * kwargs["REL_FREQ"]),
    26: lambda **kwargs : kwargs["CL"] / kwargs["REL_FREQ"]
}



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
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_ode_function(tid, t, dx, x, acc, cp, dp, sp):

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
@cuda.jit(nb.boolean(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_timesteps(tid, t, x, acc, cp, dp, sp):
    return False


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_initialization(tid, t, td, x, acc, cp, dp, sp):

    x[4] = 0.0         # Primary Bjerknes Force Integral
    x[5] = 0.0         # Drag Forca Integral
    acc[2] = x[1]      # Initial Bubble Position


@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp, dp, sp):

    # Calculate the time-averaged Bjerknes force
    dt = td[1] - td[0]
    acc[0] = x[4] / dt              # Time-Averaged Primary Bjerknes force
    acc[1] = x[5] / dt              # Time-Averaged Drag Force
    acc[3] = x[1]                   # Last Bubble Position
    acc[4] = (acc[3]-acc[2]) / dt   # Average translational velocity 

    # Increas the time domain by dt
    td[0] += 1.0 * dt
    td[1] += 1.0 * dt

