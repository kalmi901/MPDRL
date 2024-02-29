"""
This is a single-file implementation of the model KM0D.


"""


import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Optional, Callable


#  MATERIAL PROPERTIES (SI Untis)
PV  = 0.0       # Vapour Pressure [Pa]
RHO = 998.0     # Liquod Density [kg/m**3]
ST  = 0.0725    # Surface Tension [N/m]
VIS = 0.001     # Liquid viscosity [Pa s]
CL  = 1500      # Liqid Sound Speed
P0  = 1.0*1e5   # Ambient Pressure [Pa]
PE  = 1.4       # Polytrophic Exponent [-]



# Try not to modify
__AC_FUN_SIG = nb.float64(nb.float64, nb.float64[:], nb.float64[:])
__ODE_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])

def setup(k):

    global _PA, _PAT
    global _ode_function
    # ----------------- Acoustic Field -------------------

    @nb.njit(__AC_FUN_SIG, inline="always")
    def _PA(t, sp, dp):
        """
        Excitation Pressure (dual-frequency) \n
        Arguments: \n
            t (nb.float64)      - Dimensionless time
            sp(nb.float64[:])   - Static constants
            dp(nb.float64[:])   - Dynamic constant

        Returns: \n
            p (nb.float64)      - Pressure amplitude
        """

        p = 0.0
        for i in range(k):
            p += dp[i] * np.sin(2*np.pi*sp[1]*dp[k + i] * t + dp[2*k + i])

        return p

    @nb.njit(__AC_FUN_SIG, inline='always')
    def _PAT(t, sp, dp):
        """
        The time derivative of the excitation pressure \n
        Arguments: \n
            t (nb.float64)      - Dimensionless time
            sp(nb.float64[:])   - Static constants
            dp(nb.float64[:])   - Dynamic constants

        Returns: \n
            pt(nb.float64)       - time derivative of pressure amplitude
        """

        pt = 0.0
        for i in range(k):
            pt += dp[i] * dp[k + i] \
                * np.cos(2*np.pi*sp[1]*dp[k + i] * t + dp[2*k + i])

        return  pt


    # ---------------------- ODE System  -----------------------

    @nb.njit(__ODE_FUN_SIG)
    def _ode_function(t, x, cp, sp, dp):
        """
        Dimensionless Keller--Miksis Equation \n
        ___ \n
        Arguments: \n
            t (nb.float64)      -   Dimensionless time
            x (nb.float64)      -   State variables (dimless R, U)
            cp (nb.float64[:])  -   Pre-computed constants
            sp (nb.float64[:])  -   Static bubble size independent consntants
            dp (nb.float64[:])  -   Dynamic constants

        Returns: \n
            dx (nb.float64[:])  dx/dt = f(x, t)


        First-order ode System: \n

        x[0]         - Dimensionless Bubble Radius
        x[1]         - Dimensionless Bubble Wall Velocity

        dx[0] = x[1] - Dimensionless Bubble Wall Velocity
        dx[1] = N/D  - Dimensionless Bubble Wall Acceleration

        """

        dx = np.zeros_like(x)
        rx0 = 1.0 / x[0]
        p = rx0**sp[0]

        N = (cp[0] + cp[1]*x[1]) * p \
            - cp[2] * (1 + cp[7]*x[1]) - cp[3]* rx0 - cp[4]*x[1]*rx0 \
            - 1.5 * (1.0 - cp[7]*x[1] * (1.0/3.0))*x[1]*x[1] \
            - (1 + cp[7]*x[1]) * cp[5] * _PA(t, sp, dp) - cp[6] * _PAT(t, sp, dp) * x[0]

        D = x[0] - cp[7]*x[0]*x[1] + cp[4]*cp[7]
        rD = 1.0 / D

        dx[0] = x[1]
        dx[1] = N * rD

        return dx


class SingleBubble:
    """
    This object provides a simple interface to run simulations

    """

    def __init__(self,
                 R0: float,
                 FREQ: List[float],
                 PA: List[float],
                 PS: List[float] = [0.0, 0.0],
                 REF_FREQ: Optional[float] = None,
                 k: int = 1) -> None:
        
        """
        Arguments: \n
            R0       - equilibrium bubble radius (micron)
            FREQ     - excitation frequecy [f0, f1] (kHz)
            PA       - pressure amplitude [PA0, PA1] (bar)
            PS       - phase shift [PS0, PS1] (radians)
            REF_FREQ - relative frequency (optional, default FREQ[0]) (kHz)
            k        - number of harmonic components (int)
        """
        
        setup(k)        # Call Setup to initialize the model equations
        # Convert to SI Units
        self._R0 = R0 * 1e-6
        self._FREQ = np.array(FREQ, dtype=np.float64) * 1e3
        self._PA = np.array(PA, dtype=np.float64) * 1e5
        self._PS = np.array(PS, dtype=np.float64)
        self._REF_FREQ = REF_FREQ *1e3 if REF_FREQ is not None else FREQ[0]*1e3
        self._k = k


        # Initial Conditions
        self.r0 = 1.0       # Bubble Radius
        self.u0 = 0.0       # Bubble Wall Veloctiy

        # Timedomain
        self.t0 = 0.0
        self.T  = 100.0     # Number Of Periods

        self.cp = np.zeros((8, ), dtype=np.float64)
        self.sp = np.zeros((2, ), dtype=np.float64)
        self.dp = np.zeros((3*k, ), dtype=np.float64 )
        self._update_control_parameters()


    def integrate(self,
                  atol: float = 1e-9,
                  rtol: float = 1e-9,
                  min_step: float = 1e-16,
                  event: Optional[Callable] = None,
                  **options):
        
        # TODO: **options are not used
        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.T],
                        y0=[self.r0, self.u0],
                        args=(self.cp, self.sp, self.dp),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step,
                        events=event)
        
        return res.t, res.y[0], res.y[1]



    def _update_control_parameters(self):
        """
        Pre-compute the constant parameters
        """

        wr = 2.0 * np.pi * self._REF_FREQ
        
        # Keller--Miksis Constants
        self.cp[0] = (2.0 * ST / self._R0 + P0 - PV) * (2.0* np.pi / self._R0 / wr)**2.0 / RHO
        self.cp[1] = (1.0 - 3.0*PE) * (2 * ST / self._R0 + P0 - PV) * (2.0*np.pi / self._R0 / wr) / CL / RHO
        self.cp[2] = (P0 - PV) * (2.0 *np.pi / self._R0 / wr)**2.0 / RHO
        self.cp[3] = (2.0 * ST / self._R0 / RHO) * (2.0 * np.pi / self._R0 / wr)**2.0
        self.cp[4] = 4.0 * VIS / RHO / (self._R0**2.0) * (2.0* np.pi / wr)
        self.cp[5] = ((2.0 * np.pi / self._R0 / wr)**2.0) / RHO
        self.cp[6] = ((2.0 * np.pi / wr)** 2.0) / CL / RHO / self._R0
        self.cp[7] = self._R0 * wr / (2 * np.pi) / CL

        # Static Constants
        self.sp[0] = 3.0 * PE
        self.sp[1] = 1.0 / wr

        # Dynamic Constants
        for i in range(self._k):
            self.dp[          + i] = self._PA[i]                        # Pressure Amplitude
            self.dp[  self._k + i] = 2.0 * np.pi * self._FREQ[i]        # Angular Frequency
            self.dp[2*self._k + i] = self._PS[i]                        # Phase Shift


if __name__ == "__main__":

    R0 = 44.8               # (micron)
    FREQ = [31.25, 25.0]    # (kHz)
    PA   = [0.225, 0.0]     # (bar)

    km_bubble = SingleBubble(R0, FREQ, PA, k=2)
    
    t, r, u = km_bubble.integrate()

    plt.plot(t, r, "k-")
    plt.show()