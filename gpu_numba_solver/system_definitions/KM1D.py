import numba as nb
from numba import cuda
import math

# ----------------- ODE Solver Parameters ------------------
DEFAULT_SOLVER_OPTS = {
    "SD"  : 4,              # System Dimensios (r, u, x, v, Fb1, Fd)
    "NCP" : 12,             # Number of Control Parameters
    "NDP" : 4,              # Number of Dynamic Parameters
    "NSP" : 5,              # Number of Shared Parameters
    "NACC": 5,              # Number of Accessories
    "SOLVER": "RKCK45",     # ODE-solver algo
    "BLOCKSIZE": 64,        # Number of Threads per block
    "ATOL" : 1e-9,          # Absolute Tolerance
    "RTOL" : 1e-9,          # Relative Tolerance
    "NT": 1,                # Number of threads (at least on active thread requidred)
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
    "k"     : 2,                    # Number of Harmonic Components
    "R0"    : 44.8 * 1e-6,          # Equilibrium Radius (micron)
    "FREQ"  : [25.0*1e3, 50.0*1e3], # Excitation frequencies (Hz)       
    "PA"    : [0.8*1e5, 0.0*1e5],   # Pressure Amplotude (bar)
    "PS"    : [0.0, 0.0],           # Phase Shift (radians)
    "REL_FREQ" : 25.0*1e3,          # Relative Frequency (kHz)
}

# -------------- Control Parameters -----------------
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
    8 : lambda **kwargs : (0.5 * kwargs["CL"] / kwargs["REL_FREQ"] / kwargs["R0"])**2,
    9 : lambda **kwargs : 1.0 / (kwargs["RHO"] * kwargs["CL"] * kwargs["REL_FREQ"] * (2.0 * math.pi) * (kwargs["R0"]**3)),
    10: lambda **kwargs : 4.0 * math.pi / 3.0 * kwargs["R0"]**3.0,
    11: lambda **kwargs : 12.0 * math.pi * kwargs["VIS"] * kwargs["R0"],
}

SP = {
    0 : lambda **kwargs : 3 * kwargs["PE"],
    1 : lambda **kwargs : 1.0 / (2.0 * math.pi * kwargs["REL_FREQ"]),
    2 : lambda **kwargs : kwargs["CL"] / kwargs["REL_FREQ"] / (2.0 * math.pi),
    3 : lambda **kwargs : kwargs["CL"],
    4 : lambda **kwargs : 1.0 / kwargs["RHO"] / kwargs["CL"] 
}

DP = {
    0 : lambda i, **kwargs : kwargs["PA"][i],
    1 : lambda i, **kwargs : 2.0 * math.pi * kwargs["FREQ"][i],
    2 : lambda i, **kwargs : kwargs["PS"][i],
    3 : lambda i, **kwargs : 2.0 * math.pi * kwargs["FREQ"][i] / kwargs["CL"]
}


__AC_FUN_SIG = nb.float64(nb.float64, nb.float64, nb.float64[:], nb.float64[:])
def setup(k=DEFAULT_EQ_PROPS["k"], ac_field="CONST"):
    global _PA, _PAT, _GRADP, _UAC
    global per_thread_ode_function

    print("Initialize the ode system")
    print(f"Number of harmonoc compoentes: {k}")
    print(f"Acoustic field type: {ac_field}")

    # ----------------- ACOUSTIC FIELD ------------------------
    if ac_field == "CONST":
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PA(t, x, sp, dp):
            p = 0.0
            for i in range(k):
                p += dp[i] * math.sin(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return p


        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PAT(t, x, sp, dp):
            pt = 0.0
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                            * math.cos(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])
                
            return pt

        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _GRADP(t, x, sp, dp):
            return 0.0

        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _UAC(t, x, sp, dp):
            return 0.0
    
    elif ac_field == "SW_A":
        # TODO!!!!!!!!
        pass

    elif ac_field == "SW_N":
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PA(t, x, sp, dp):
            p = 0.0
            for i in range(k):
                p += dp[i]  * math.sin(2*math.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                            * math.sin(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])
            return p
        
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PAT(t, x, sp, dp):
            pt = 0.0
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                           * math.sin(2*math.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * math.cos(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])
            return pt
                
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _GRADP(t, x, sp, dp):
            px = 0.0
            for i in range(k):
                px+= dp[i] * dp[3*k + i] \
                           * math.cos(2*math.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * math.sin(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i]) 
            return px
        
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _UAC(t, x, sp, dp):
            ux = 0.0
            for i in range(k):
                ux+=-dp[i] * sp[4] \
                           * math.cos(2*math.pi*sp[2]*dp[3*k +i] * x + dp[2*k + i]) \
                           * math.cos(2*math.pi*sp[1]*dp[  k +i] * t + dp[2*k + i])
            return ux

    else:
        print("Acoustic field type is not correct!")
        exit()


    # ------------------- ODE Functions -----------------------
    @cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
    def per_thread_ode_function(tid, t, dx, x, acc, cp, dp, sp):
        rx0 = 1.0 / x[0]
        p = rx0**sp[0]

        N = (cp[0] + cp[1]*x[2]) * p \
                - cp[2] * (1 + cp[7]*x[2]) - cp[3]* rx0 - cp[4]*x[2]*rx0 \
                - 1.5 * (1.0 - cp[7]*x[2] * (1.0/3.0))*x[2]*x[2] \
                - (1 + cp[7]*x[2]) * cp[5] * _PA(t, x[1], sp, dp) - cp[6] * _PAT(t, x[1], sp, dp) * x[0] \
                + cp[8] * x[3]*x[3]                                       # Feedback term

        D = x[0] - cp[7]*x[0]*x[2] + cp[4]*cp[7]
        rD = 1.0 / D

        Fb1 = - cp[10]*x[0]*x[0]*x[0] * _GRADP(t, x[1], sp, dp)           # Primary Bjerknes Force
        Fd  = - cp[11]*x[0] * (x[3]*sp[3] - _UAC(t, x[1], sp, dp) )       # Drag Force

        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = N * rD
        dx[3] = 3*(Fb1+Fd)*cp[9]*rx0*rx0*rx0 - 3.0*x[2]*rx0*x[3]


# --------------------- ACCESSORIES --------------------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp, dp, sp):
    dt = td[1] - td[0]
    # Increase the time domain by dt
    td[0] += 1.0 * dt
    td[1] += 1.0 * dt
