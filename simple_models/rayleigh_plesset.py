"""
TODO
"""


import numpy as np
import numba as nb
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Optional, Callable

# MATERIAL PROPERTIES
PV = 3.166775638952003e+03;         # Vapour Pressures [PA]
RHO= 9.970639504998557e+02;         # Liquid Density [kg/m**3]
ST = 0.071977583160056;             # Surface Tension [N/m]
VIS= 8.902125058209557e-04          # Liquid Viscosity [Pa s]
CL = 1.497251785455527e+03          # Liquid Sound Speed [m/s]
P0 = 1.0                            # Ambient Pressure [bar]
PE = 1.4                            # Polytrophic Exponent [-]



# Try not to modify
__ODE_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:])



@nb.njit(__ODE_FUN_SIG)
def _ode_function(t, x, cp):
    """
    Dimensionless Rayleigh--Plesset Equation \n
    ___ \n
    Parameters: \n
        t (float64):    - Time \n
        x (float64[:])  - State Vaariables \n
        cp (float64[:]) - Control Parameters \n

    Retursn: \n
        dx (float64[:]) - dx/dy = f(x, t)

    ODE System ----- \n

    x[0]            - Bubble Radius \n
    x[1]            - Bubble Wall velocity \n

    dx[0] = x[1]    - Bubble Wall velocity \n
    dx[1] = RP(x,t) - Bubble Wall acceleration

    """

    dx = np.zeros_like(x)
    rx0 = 1.0 / x[0]

    dx[0] = x[1]
    dx[1] =  cp[0] + cp[1] * rx0**cp[2] - cp[3] * rx0 - cp[4] * x[1] * rx0 - cp[5] * np.sin(2*np.pi*t) - 1.5 * x[1] * x[1] * rx0

    return dx



class SingleBubble:


    def __init__(self,
                 R0: float,
                 FREQ: float,
                 PA: float) -> None:
        
        self._FREQ  = FREQ
        self._PA    = PA
        self._R0    = R0

        # Initial Conditions
        self.r0     = 1.0   # Initial Bubble Radius
        self.u0     = 0.0   # Initial Bubble Wall velocity

        self.t0     = 0.0
        self.T      = 100.0 # Number of periods

        self.cp = np.zeros((6, ), dtype=np.float64)
        self._update_control_parameters()


    def integrate(self,
                  atol: float = 1e-9,
                  rtol: float = 1e-9,
                  min_step: float = 1e-16,
                  event: Callable = None,
                  **options):


        if event is not None:
            # Hack event detection, to avoid stopping on the initial condition
            res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.t0+1e-6],
                        y0=[self.r0, self.u0],
                        args=(self.cp, ),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step)
            
            self.r0 = res.y[0][-1]
            self.u0 = res.y[1][-1]


        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0+1e-6, self.T],
                        y0=[self.r0, self.u0],
                        args=(self.cp, ),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step,
                        events=event)



        return res.t, res.y[0], res.y[1]

    # Private Methods
    def _update_control_parameters(self):

        # Convert To SI Units
        w       = self._FREQ * 1000 * 2 * np.pi
        R0      = self._R0 * 1e-6
        Pinf    = P0 * 1e5
        Pa      = self._PA * 1e5

        self.cp[0] = (PV - Pinf) / RHO * (2 * np.pi / w / R0)**2
        self.cp[1] = (2.0 * ST / R0 + Pinf - PV) / RHO * (2 * np.pi / w / R0)**2
        self.cp[2] = 3.0 * PE
        self.cp[3] = 2.0 * ST / RHO / R0 * (2 * np.pi / w / R0)**2
        self.cp[4] = 4.0 * VIS / RHO / (R0**2) * (2 * np.pi / w)
        self.cp[5] = Pa / RHO * (2 * np.pi / w / R0)**2



def max_radius_event(t, x, cp): return x[1]
max_radius_event.terminal = False
max_radius_event.direction = -1

if __name__ == "__main__":
    R0 = 44.8
    PA = 0.225
    FREQ = 31.25

    rp_bubble = SingleBubble(R0, FREQ, PA)

    t, r, u = rp_bubble.integrate(event=max_radius_event)

    
    pd.DataFrame(np.hstack((t[:, np.newaxis], r[:, np.newaxis])),
                dtype=np.float64).to_csv("RP_Solution_Debug.csv", header=False, index=False)

    plt.plot(t, r, "k-")
    plt.show()