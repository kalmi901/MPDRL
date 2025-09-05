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
    "NCT" : 8,              # Number of Coupling Terms
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

    from _coupled_system_definition_template import __ODE_FUN_SIGNATURE, __MAT_VEC_SIGNATURE
   
    #print(__ODE_FUN_SIGNATURE)
    print("Initialize the ode system")
    print(f"Number of harmonoc compoentes: {k}")
    print(f"Acoustic field type: {ac_field}")
    _PA, _PAT, _GRADP, _UAC = AC1D.setup(ac_field, k)

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_initialization(ups, gsid, luid, t, td, x, cpf, up, gp, sp, dp, cpt, mx):
        
        if luid == 0:
            print("per_block_initialization")

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_finalization(ups, gsid, luid, t, td, x, cpf, up, gp, sp, dp, cpt, mx):
        
        if luid == 0:
            print("per_block_finalization")

    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_ode_function(ups, gsid, luid, t, dx, x, cpf, up, gp, sp, dp, cpt, mx):
        """
        RHS of the ODE without the coupling terms
        ups  - unity per system
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
        cpf[0] = rD                         # G0
        cpf[1] = x[3] * rD                  # G1 
        cpf[2] = x[2] * rx0                 # G2
        cpf[3] = rx0 * rx0                  # G3

        # Coupling Terms
        cpt[0, luid] = x[0] * x[2]**2  
        cpt[1, luid] = x[0]**2 * x[2]
        cpt[2, luid] = x[0]**2 * x[2] * x[3]
        cpt[3, luid] = x[0]**3 * x[3]
        cpt[4, luid] = x[0]**3 * x[3]**2
        cpt[5, luid] = x[0]**2
        cpt[6, luid] = x[0]**3
        cpt[7, luid] = x[1]               # Bubble positions

        #print(gsid)

        if (luid == 0):
            #print(dx[0])
            #print(dx[1])
            #print(dx[2])
            #print(dx[3])
            pass
            # Debug print


    @cuda.jit(__ODE_FUN_SIGNATURE, device=True, inline=True)
    def per_block_explicit_coupling(ups, gsid, luid, t, dx, x, cpf, up, gp, sp, dp, cpt, mx):
        """
        Correction of the RHS including the explicit coupling terms
        Note: per_block_ode_function is called first, coupling factors anc coupling terms are pre-calculated
        cpt[7]: contains the dimensionless bubble coordinate
        """
        
        # --- Radial Couplings ---
        # First-order
        xi = cpt[7, luid]                           # Local Bubble Positin

        sum_rad_g0 = 0.0
        sum_rad_g1 = 0.0
        sum_trn_ng = 0.0
        sum_trn_g2 = 0.0
        sum_trn_g3 = 0.0

        # Matrix - Vector Products 
        for j in range(ups):
            # -- Coupling Terms --
            h0 = cpt[0, j]
            h1 = cpt[1, j]
            h2 = cpt[2, j]
            h3 = cpt[3, j]
            h4 = cpt[4, j]

            # -- Calculate the distance and the direction --
            xj = cpt[7, j]                          # Coupled Bubble Positions
            delta   = math.fabs(xi - xj)
            m       = 1.0 - nb.float64(luid == j)   # Diagonal Mask 1 ha nem diagonál, 0 ha diagonál (nincs branch)
            r_delta = m / max(delta, 1e-30)         # inv = m / max(a, eps)  → diagonálon inv=0, különben 1/|xi-xj|
            s = nb.float64((luid>j) - (j>luid))
            r_delta2 = r_delta * r_delta
            sr_delta2= r_delta2 * s
            r_delta3 = r_delta2*r_delta

            # -- Coupling matrices --
            m0 = mx[0, luid, j]   # radial
            m1 = mx[1, luid, j]   # translational
            m2 = mx[2, luid, j]   # liquid velocity

            sum_rad_g0 += m0 * (
                -2.0 * r_delta   * h0
                -2.5 * sr_delta2 * h2
                -1.0 * r_delta3  * h4
            )

            sum_rad_g1 += m0 * (
                -0.5 * sr_delta2 * h2
                -0.5 * r_delta3  * h3
            )

            sum_trn_ng += m1 * (
                 2.0 * sr_delta2 * h0
                +5.0 * r_delta3  * h2
            )

            sum_trn_g2 += m1 * (
                    sr_delta2 * h1
                +   r_delta3  * h3
            )

            sum_trn_g3 += m2 * (
                    sr_delta2 * h1
                +   r_delta3  * h3
            )

        # Correct RHS
        dx[2] += sum_rad_g0 * cpf[0] + sum_rad_g1 * cpf[1]
        dx[3] += sum_trn_ng + sum_trn_g2 * cpf[2] + sum_trn_g3 * cpf[3]


    @cuda.jit(__MAT_VEC_SIGNATURE, device=True, inline=True)
    def per_block_implicit_mat_vec(ups, gsid, luid, cpf, cpt, mx, v, Av):
        
        xi = cpt[7, luid]            # Local Bubble Position
        av_rad = 0.0
        av_trn = 0.0

        for j in range(ups):
            # --- Coupling Terms ---
            h5 = cpt[5, j]
            h6 = cpt[6, j]

            # --- Calculate the distantence and the direction ---
            xj    = cpt[7, j]                       # Coupled bubble position
            delta = math.fabs(xi - xj)              # Dimensionless Distance
            m     = 1.0 - nb.float64(luid == j)     # Diagonal Mask
            r_delta = m / max(delta, 1e-30)         # Avoid division by 0
            s = nb.float64((luid>j) - (j>luid))     # sign(d)
            r_delta2 = r_delta * r_delta
            sr_delta2 = r_delta2 * s
            r_delta3 = r_delta2 * r_delta

            # -- Coupling matrices --
            m0 = mx[0, luid, j]   # radial
            m1 = mx[1, luid, j]   # translational
            vr = v[0, j]          # v[:num_bubbles]
            vt = v[1, j]          # v[num_bubbles:]

            # TODO: implement calculation here
            av_rad += m0 * (
                    r_delta   * h5 * vr
                +   sr_delta2 * h6 * vt
            )

            av_trn += -m1 * (
                    sr_delta2 * h5 * vr
                +   r_delta3  * h6 * vt
            )

        # Add Identity Av + Iv
        Av[0, luid] = av_rad * cpf[0] + v[0, luid]
        Av[1, luid] = av_trn + v[1, luid]


        
    functions = {
        "per_block_ode_function"      : per_block_ode_function,
        "per_block_initialization"    : per_block_initialization,
        "per_block_finalization"      : per_block_finalization,
        "per_block_explicit_coupling" : per_block_explicit_coupling,
        "per_block_implicit_mat_vec"  : per_block_implicit_mat_vec
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