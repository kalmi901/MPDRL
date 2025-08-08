import numba as nb
from numba import cuda
import math
from .system_registry import ode_model

# ----------------- ODE Solver Parameters ------------------
DEFAULT_SOLVER_OPTS = {
    "NS"  : 1,              # Number of systems (at least on active system requidred)
    "SPB" : 1,              # Number of systems per block
    "UPS" : 4,              # Number of units per system
    "UD"  : 4,              # Unit System Dimension (R, U, x, v)
    "NUP" : 13,             # Number of Unit Control Parameters
    "NSP" : 0,              # Number of System Parameters (different from system to system, shared by all units per system)
    "NGP" : 5,              # Number of Global Parameters (shared by all systems)
    "NDP" : 4,              # Number of Dynamic parameter (later these params will be adjusted by the RL-Agent, diffretn from system to system, shared by all units)
    "NUA" : 0,              # Number of Unit Accessories
    "NSA" : 0,              # Number of System Accessories
    "NC"  : 3,              # Number of Coupling Matrices
    "NCT" : 6,              # Number of Coupling Terms
    "NCF" : 4,              # Number of Coupling Factor
    "NE"  : 1,              # Number of Events
    "SOLVER": "RKCK45",     # ODE-solver algo
    "BLOCKSIZE": 64,        # Number of Threads per block
    "ATOL" : 1e-9,          # Absolute Tolerance
    "RTOL" : 1e-9,          # Relative Tolerance
    "ETOL": 1e-6,           # Event Tolerance
    "EDIR":-1,              # Event Direction
    "NDO" : 1000            # Numberf of Dense Output
}

# -------------- Global Model Constants Parameters ----------
# Material Properties
DEFAULT_MAT_PROPS = {
    "PV"  : 0.0,    # Vapour Pressure [Pa]
    "RHO" : 998.0,  # Liquid Density  [kg/m**3]  
    "ST"  : 0.0725, # Surface Tension [N/m]
    "VIS" : 0.001,  # Liquid Viscosity [Pa s]
    "CL"  : 1500,   # Liquid Sound Speed [m/s]
    "P0"  : 1.013e5,# Ambient Pressure [Pa]
    "PE"  : 1.4,    # Polytrophic Exponent
}

# Equation Properties (Default values in SI Units
DEFAULT_EQ_PROPS = {
    "k"     : 2,                        # Number of Harmonic Components
    "R0"    : [50*1e-6] * 128,          # Equilibrium Radius (micron) (len(...)=NUP)
    "FREQ"  : [25.0*1e3, 50.0*1e3],     # Excitation frequencies (Hz)       
    "PA"    : [0.8*0e5, 0.0*1e5],       # Pressure Amplotude (bar)
    "PS"    : [0.0, 0.0],               # Phase Shift (radians)
    "REL_FREQ" : 25.0*1e3,              # Relative Frequency (kHz)
}

# ---------- CONTROL PARAMETERS -------
# Unit Parameters -----
UP = {
    0 : lambda i, **kwargs : (2.0 * kwargs["ST"] / kwargs["R0"][i] + kwargs["P0"] - kwargs["PV"]) 
                            * (1.0 / kwargs["R0"][i] / kwargs["REL_FREQ"])**2.0 / kwargs["RHO"],
    1 : lambda i, **kwargs : (1.0 - 3.0*kwargs["PE"]) * (2 * kwargs["ST"] / kwargs["R0"][i] + kwargs["P0"]- kwargs["PV"]) 
                            * (1.0 / kwargs["R0"][i] / kwargs["REL_FREQ"]) / kwargs["CL"] / kwargs["RHO"],
    2 : lambda i, **kwargs : (kwargs["P0"] - kwargs["PV"]) * (1.0 / kwargs["R0"][i] / kwargs["REL_FREQ"])**2.0 / kwargs["RHO"], 
    3 : lambda i, **kwargs : (2.0 * kwargs["ST"] / kwargs["R0"][i]/ kwargs["RHO"]) * (1.0 / kwargs["R0"][i] / kwargs["REL_FREQ"])**2.0,
    4 : lambda i, **kwargs : 4.0 * kwargs["VIS"] / kwargs["RHO"] / (kwargs["R0"][i]**2.0) * (1.0 / kwargs["REL_FREQ"]),
    5 : lambda i, **kwargs : ((1.0 / kwargs["R0"][i] / kwargs["REL_FREQ"])**2.0) / kwargs["RHO"],
    6 : lambda i, **kwargs : ((1.0 / kwargs["REL_FREQ"])** 2.0) / kwargs["CL"] / kwargs["RHO"] / kwargs["R0"][i],
    7 : lambda i, **kwargs : kwargs["R0"][i] * kwargs["REL_FREQ"] / kwargs["CL"],
    8 : lambda i, **kwargs : (0.5 * kwargs["CL"] / kwargs["REL_FREQ"] / kwargs["R0"][i])**2,
    9 : lambda i, **kwargs : 1.0 / (kwargs["RHO"] * kwargs["CL"] * kwargs["REL_FREQ"] * (2.0 * math.pi) * (kwargs["R0"][i]**3)),
    10: lambda i, **kwargs : 4.0 * math.pi / 3.0 * kwargs["R0"][i]**3.0,
    11: lambda i, **kwargs : 12.0 * math.pi * kwargs["VIS"] * kwargs["R0"][i],
    12: lambda i, **kwargs : kwargs["R0"][i] / ((kwargs["CL"] / kwargs["REL_FREQ"]))
}

# Global Parameters ----
GP = {
    0 : lambda **kwargs : 3 * kwargs["PE"],
    1 : lambda **kwargs : 1.0 / (2.0 * math.pi * kwargs["REL_FREQ"]),
    2 : lambda **kwargs : kwargs["CL"] / kwargs["REL_FREQ"] / (2.0 * math.pi),
    3 : lambda **kwargs : kwargs["CL"],
    4 : lambda **kwargs : 1.0 / kwargs["RHO"] / kwargs["CL"] 
}

# Dynamic (System) Parameters ---
DP = {
    0 : lambda i, **kwargs : kwargs["PA"][i],
    1 : lambda i, **kwargs : 2.0 * math.pi * kwargs["FREQ"][i],
    2 : lambda i, **kwargs : kwargs["PS"][i],
    3 : lambda i, **kwargs : 2.0 * math.pi * kwargs["FREQ"][i] / kwargs["CL"]
}

# Coupling Matrix (Shared) --- 
CM = {
    0 : lambda i, j, **kwargs : kwargs["R0"][j]**3 / kwargs["R0"][i]**2 / (kwargs["CL"] / kwargs["REL_FREQ"]),
    1 : lambda i, j, **kwargs : 3.0 * (kwargs["R0"][j] / (kwargs["CL"] / kwargs["REL_FREQ"]))**3,
    2 : lambda i, j, **kwargs : (18 * kwargs["VIS"] / kwargs["RHO"] / kwargs["REL_FREQ"] ) * (kwargs["R0"][j] / (kwargs["CL"] / kwargs["REL_FREQ"]))**3 / kwargs["R0"][i]**2
}


def setup(k, ac_field):
    from . import AC1D
    #global _PA, _PAT, _GRADP, _UAC
    #global per_block_ode_function

    from _coupled_system_definition_template import __ODE_FUN_SIGNATURE
   
    print(__ODE_FUN_SIGNATURE)
    print("Initialize the ode system")
    print(f"Number of harmonoc compoentes: {k}")
    print(f"Acoustic field type: {ac_field}")
    _PA, _PAT, _GRADP, _UAC = AC1D.setup(ac_field, k)

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_initialization(gsid, luid, t, td, x, cpf, up, gp, sp, dp):
        
        if luid == 0:
            print("per_block_initialization")

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_finalization(gsid, luid, t, td, x, cpf, up, gp, sp, dp):
        
        if luid == 0:
            print("per_block_finalization")

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_ode_function(gsid, luid, t, dx, x, cpf, up, gp, sp, dp):
        """
        RHS of the ODE without the coupling terms
        gsid - Global System id
        luid - Local Unit ID
        dx   - Explicit derivative terms! (Threads do not communicate here)
        """

        rx0 = 1.0 / x[0]
        p = rx0**gp[0]

        N = (up[0] + up[1]*x[2]) * p \
                - up[2] * (1 + up[7]*x[2]) - up[3]* rx0 - up[4]*x[2]*rx0 \
                - 1.5 * (1.0 - up[7]*x[2] * (1.0/3.0))*x[2]*x[2] \
                - (1 + up[7]*x[2]) * up[5] * _PA(t, x[1], gp, dp) - up[6] * _PAT(t, x[1], gp, dp) * x[0] \
                + up[8] * x[3]*x[3]
        
        D = x[0] - up[7]*x[0]*x[2] + up[4]*up[7]
        rD = 1.0 / D

        Fb1 = - up[10]*x[0]*x[0]*x[0] * _GRADP(t, x[1], gp, dp)             # Primary Bjerknes Force
        Fd  = - up[11]*x[0] * (x[3]*gp[3] - _UAC(t, x[1], gp, dp))          # vij is inculdued as coupling term
        du = 3.0*((Fb1+Fd)*up[9]*rx0*rx0 - x[2]*x[3]) * rx0

        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = N * rD
        dx[3] = du   

        # Share us pleeease :)
        # position x[1]  -> for distance
        # coupling terms j
        # coupling factors i
        # Coupling factors
        cpf[0] = rD
        cpf[1] = x[3] * rD
        cpf[2] = x[2] * rx0
        cpf[3] = rx0 * rx0

        #print(gsid)

        if (luid == 0):
            #print(dx[0])
            #print(dx[1])
            #print(dx[2])
            #print(dx[3])
            pass
            # Debug print


    def per_block_explicit_coupling(gsid, luid, dx, x):
        """
        Correction of the RHS including the explicit coupling terms
        Note: per_bloc_ode_function is called first, 
        """

        
    functions = {
        "per_block_ode_function"   : per_block_ode_function,
        "per_block_initialization" : per_block_initialization,
        "per_block_finalization"   : per_block_finalization
    }

    return functions


# --- Register ODE MODEL ----
@ode_model("KM1DNBC")
class KM1DNBC:

    description = f"Keller--Miksis equation 1-Dimensional N-Bubble Coupled System"

    defaults = {
        "mat_props"   : DEFAULT_MAT_PROPS,
        "eq_props"    : DEFAULT_EQ_PROPS,
        "solver_opts" : DEFAULT_SOLVER_OPTS
    }

    parameters = {
        "UP" : UP,          # Unit Parameters
        "DP" : DP,          # Dynamic Parameters
        "GP" : GP,          # Global Parameters 
        "CM" : CM,          # Constant Copuling Matrix
    }

    functions = {}

    @classmethod
    def setup(cls,**kwargs):
        cls.functions = setup(**kwargs)