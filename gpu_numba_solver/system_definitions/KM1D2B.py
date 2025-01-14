import numba as nb
from numba import cuda
import math
from . import AC1D

# ----------------- ODE Solver Parameters ------------------
DEFAULT_SOLVER_OPTS = {
    "SD"  : 8,              # System Dimensios (r, u, x, v, Fb1, Fd)
    "NCP" : 32,             # Number of Control Parameters
    "NDP" : 4,              # Number of Dynamic Parameters
    "NSP" : 5,              # Number of Shared Parameters
    "NACC": 5,              # Number of Accessories
    "NE"  : 1,              # Number of Event
    "SOLVER": "RKCK45",     # ODE-solver algo
    "BLOCKSIZE": 64,        # Number of Threads per block
    "ATOL" : 1e-9,          # Absolute Tolerance
    "RTOL" : 1e-9,          # Relative Tolerance
    "NT": 1,                # Number of threads (at least on active thread requidred)
    "ETOL" : 1e-6,          # Event Tolerance
    "EDIR" : 0              # Event Direction
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
    "R0"    : [44.8*1e-6, 44.8*1e-6],   # Equilibrium Radius (micron)
    "FREQ"  : [25.0*1e3, 50.0*1e3],     # Excitation frequencies (Hz)       
    "PA"    : [0.8*0e5, 0.0*1e5],       # Pressure Amplotude (bar)
    "PS"    : [0.0, 0.0],               # Phase Shift (radians)
    "REL_FREQ" : 25.0*1e3,              # Relative Frequency (kHz)
}


# -------------- Control Parameters -----------------
CP = {
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
    12: lambda i, **kwargs : (kwargs["R0"][1-i] / (kwargs["CL"] / kwargs["REL_FREQ"]))**3 * kwargs["CL"],
    13: lambda i, **kwargs : kwargs["R0"][1-i]**3 / kwargs["R0"][i]**2 / (kwargs["CL"] / kwargs["REL_FREQ"]),
    14: lambda i, **kwargs : 3 * (kwargs["R0"][1-i] / (kwargs["CL"] / kwargs["REL_FREQ"]))**3,
    15: lambda i, **kwargs : kwargs["R0"][i] / ((kwargs["CL"] / kwargs["REL_FREQ"]))
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


#__AC_FUN_SIG = nb.float64(nb.float64, nb.float64, nb.float64[:], nb.float64[:])
def setup(k=DEFAULT_EQ_PROPS["k"], ac_field="CONST"):
    global _PA, _PAT, _GRADP, _UAC
    global per_thread_ode_function

    print("Initialize the ode system")
    print(f"Number of harmonoc compoentes: {k}")
    print(f"Acoustic field type: {ac_field}")
    _PA, _PAT, _GRADP, _UAC = AC1D.setup(ac_field, k)
    """
    # ----------------- ACOUSTIC FIELD ------------------------
    k = DEFAULT_EQ_PROPS["k"]
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
    """

    # ------------------- ODE Functions -----------------------
    @cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
    def per_thread_ode_function(tid, t, dx, x, acc, cp, dp, sp):
        
        rd = 1.0 / abs(x[3]-x[2])                       # Inverse distance
        s  = cuda.local.array((2, ), dtype=nb.float64)
        b = cuda.local.array((4, ),  dtype=nb.float64)
        A = cuda.local.array((4, 4), dtype=nb.float64)
        for i in range(4):
            for j in range(4):
                A[i,j] = 0.0

        s[0] = 1.0
        s[1] =-1.0
        for i in range(2):
            x0i = x[i]
            x1i = x[2 + i]
            x2i = x[4 + i]
            x3i = x[6 + i]


            # Reversed
            x0j = x[1 - i]
            x2j = x[5 - i]
            x3j = x[7 - i]

            rx0 = 1.0 / x0i
            p = rx0**sp[0]
            n_cp = i * 16         # Offset in CP

            # Radial Oscillation
            N = (cp[0+n_cp] + cp[1+n_cp]*x2i) * p \
                    - cp[2+n_cp] * (1 + cp[7+n_cp]*x2i) - cp[3+n_cp]* rx0 - cp[4+n_cp]*x2i*rx0 \
                    - 1.5 * (1.0 - cp[7+n_cp]*x2i * (1.0/3.0))*x2i*x2i \
                    - (1 + cp[7+n_cp]*x2i) * cp[5+n_cp] * _PA(t, x1i, sp, dp) - cp[6+n_cp] * _PAT(t, x1i, sp, dp) * x0i \
                    + cp[8+n_cp] * x3i*x3i \
                    + cp[13+n_cp] * rd * (-2*x0j*x2j**2                                      
                                + 0.5 * rd*x0j**2 
                                * ( s[i]*(x3i*x2j + x2j*x3j) 
                                - rd*x0j*x3j*(x3i + 2*x3j) ))

            D = x0i - cp[7+n_cp]*x0i*x2i + cp[4+n_cp]*cp[7+n_cp]
            rD = 1.0 / D

            # Radial Coupling
            tmp = cp[13+n_cp] * x0j*x0j*rd*rD
            A[i,   i] = 1.0
            A[i, 1-i] = tmp                                  # i = 0;    A[0, 1]     i = 1; A[1, 0]
            A[i, 3-i] =-tmp * 0.5 *s[i] * x0j * rd           # i = 0;    A[0, 3]     i = 1; A[1, 2]
            b[i]      = N * rD

            # Translational motion
            vj = cp[12+n_cp] * x0j*x0j * rd*rd * (-s[i]*x2j + x0j*x3j*rd)                    
            Fb1 = - cp[10+n_cp]*x0i*x0i*x0i * _GRADP(t, x1i, sp, dp)                         # Primary Bjerknes Force
            Fd  = - cp[11+n_cp]*x0i * (x3i*sp[3] - _UAC(t, x1i, sp, dp) - vj)                # Drag Force

            du = 3*(Fb1+Fd)*cp[9+n_cp]*rx0*rx0*rx0 - 3.0*x2i*rx0*x3i \
                + cp[14+n_cp] *rd*rd*x0j * (-s[i]*x2j * (x0j*x2i + 2*x0i*x2j)
                                    +x0j*rd*x3j * (x0j*x2i + 5*x0i*x2j) ) * rx0
            
            # Translational Coupling
            tmp = cp[14+n_cp] * x0i * x0j*x0j * rd*rd * rx0
            A[i+2,i+2] = 1.0
            A[i+2,1-i] = tmp * s[i]                          # i = 0;    A[2, 1]    i = 1;   A[3; 0]
            A[i+2,3-i] =-tmp * x0j*rd                        # i = 0;    A[2, 3]    i = 1;   A[3, 2]
            b[2+i]     = du

        # Gauss Elimination
        for i in range(4):
            for j in range(i + 1, 4):
                c = -A[j,i] / A[i,i]
                b[j] += c * b[i]
                A[j,i] = 0.0
                for k in range(i+1,4):
                    A[j,k] += c * A[i,k]

        # Backward substitution
        y = cuda.local.array((4, ), dtype=nb.float64)
        for i in range(3, -1, -1):
            y[i] = b[i] / A[i,i]
            for j in range(i-1, -1, -1):
                b[j] -= A[j,i] * y[i]


        # Copy Back derivatives
        for i in range(8):
            if i  < 4:
                dx[i] = x[i+4]
            else:
                dx[i] = y[i-4] 


# --------------------- ACCESSORIES --------------------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_finalization(tid, t, td, x, acc, cp, dp, sp):
    # Increase the time domain by dt
    dt = td[1] - td[0]
    td[0] += 1.0 * dt
    td[1] += 1.0 * dt


# -------------- EVENTS ---------------
@cuda.jit(nb.void(nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_event_function(tid, t, ev, x, acc, cp, dp, sp):
    # Collision (Radius overlaps, 1.25 safety threshold to avoid numerical errors.  )
    ev[0] = (x[0]*cp[15] + x[1]*cp[31])*1.5  - abs(x[3] - x[2])
    #print(ev[0])

@cuda.jit(nb.boolean(nb.int32, nb.int32, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), device=True, inline=True)
def per_thread_action_after_event_detection(tid, idx, t, td, x, acc, cp, dp, sp):
    return True # Terminate Event!