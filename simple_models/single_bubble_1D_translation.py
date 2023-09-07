"""
TODO
"""


import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Optional


#  MATERIAL PROPERTIES 
PV  = 0.0       # Vapour Pressure [Pa]
RHO = 998.0     # Liquod Density [kg/m**3]
ST  = 0.0725    # Surface Tension [N/m]
VIS = 0.001     # Liquid viscosity [Pa s]
CL  = 1500      # Liqid Sound Speed
P0  = 1.0       # Ambient Pressure [bar]
PE  = 1.4       # Polytrophic Exponent [-]




# Try not to modify
__AC_FUN_SIG = nb.float64(nb.float64, nb.float64, nb.float64[:])
__ODE_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:])


def setup(ac_field):
    """
    Initialize the acoustic field and the ODE System
    setup() is called automatically when a "SingleBubble" object is created
    """

    global _PA, _PAT, _GRADP, _UAC
    global _ode_function

    # -----------------------------------------
    # ---------- Acoustic Field ---------------
    # -----------------------------------------

    if ac_field == "CONST":
        @nb.njit(__AC_FUN_SIG, inline="always")
        def _PA(t, x, cp):
            return cp[9] * np.sin(2*np.pi*t*cp[11]*cp[25]) \
                + cp[10] * np.sin(2*np.pi*t*cp[12]*cp[25] + cp[13])
        
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, cp):
            return  cp[9]  * cp[11] * np.cos(2*np.pi*t*cp[11]*cp[25]) \
                  + cp[10] * cp[12] * np.cos(2*np.pi*t*cp[12]*cp[25] + cp[13])

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, cp):
            return 0.0

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, cp):
            return 0.0
        
    elif ac_field == "SW":
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PA(t, x, cp):
            return cp[9]  * np.sin(2*np.pi*x*cp[26]*cp[21]) * np.sin(2*np.pi*t*cp[11]*cp[25]) \
                +  cp[10] * np.sin(2*np.pi*x*cp[26]*cp[22]) * np.sin(2*np.pi*t*cp[12]*cp[25] + cp[13])

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, cp):
            return cp[9]  * cp[11] * np.sin(2*np.pi*x*cp[26]*cp[21]) * np.cos(2*np.pi*t*cp[11]*cp[25]) \
                 + cp[10] * cp[12] * np.sin(2*np.pi*x*cp[26]*cp[22]) * np.cos(2*np.pi*t*cp[12]*cp[25] + cp[13])

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, cp):
            return cp[9]  * cp[19] * np.cos(2*np.pi*x*cp[26]*cp[21]) * np.sin(2*np.pi*t*cp[11]*cp[25]) \
                 + cp[10] * cp[20] * np.cos(2*np.pi*x*cp[26]*cp[22]) * np.sin(2*np.pi*t*cp[12]*cp[25] + cp[13] )

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, cp):
            return cp[9]  * cp[23] * np.cos(2*np.pi*x*cp[26]*cp[21]) * np.cos(2*np.pi*t*cp[11]*cp[25]) \
                 + cp[10] * cp[23] * np.cos(2*np.pi*x*cp[26]*cp[22]) * np.cos(2*np.pi*t*cp[12]*cp[25] + cp[13] )
        
    else:
        print(f"Error: Acoustic field type \"{ac_field}\" is not implemented!")
        exit()

    @nb.njit(__ODE_FUN_SIG)
    def _ode_function(t, x, cp):
        """
        Dimensionless Keller--Miksis equation + Translational bubble motion \n
        ___\n
        Parameters: \n
            t (float64):    - Time \n
            x (float64[:])  - State Variables \n
            cp (float64[:]) - Control Parameters \n

        Returns: \n
            dx (float64[:])
            
        ODE Systems ------

        x[0] - Bubble Radius \n
        x[1] - Bubble Position \n
        x[2] - Bubble Wall Velocity \n
        x[3] - Bubble Translational Velocity \n

        dx = f(x) \n

        dx[0] = x[2] --> Bubble Wall velocity \n
        dx[1] = x[3] --> Bubble Translational velocity \n
        dx[2] = N/D  --> Keller--Miksis equation \n
        dx[3] = N2   --> Equation of motion \n
        
        """


        dx = np.zeros_like(x, dtype = np.float64)
        rx1 = 1.0 / x[0]
        p = rx1**cp[8]

        N = (cp[0] + cp[1]*x[2]) * p \
        - cp[2] * (1 + cp[7]*x[2]) - cp[3]* rx1 - cp[4]*x[2]*rx1 \
        - 1.5 * (1.0 - cp[7]*x[2] * (1.0/3.0))*x[2]*x[2] \
        - (1 + cp[7]*x[2]) * cp[5] * _PA(t, x[1], cp) - cp[6] * _PAT(t, x[1], cp) * x[0] \
        + 0.25 * x[3]*x[3]*cp[15]                                        # Feedback from translational motion

        D = x[0] - cp[7]*x[0]*x[2] + cp[4]*cp[7]
        
        rD = 1.0 / D

        Fb1 = - cp[17]*x[0]*x[0]*x[0] * _GRADP(t, x[1], cp)            # Primary Bjerknes Force
        Fd  = - cp[18]*x[0] * (x[3]*cp[24] - _UAC(t, x[1], cp) )       # Drag Force

        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = N*rD
        dx[3] = 3*(Fb1+Fd)*cp[16]*rx1*rx1*rx1 - 3.0*x[2]*rx1*x[3]

        return dx


class SingleBubble:

    # Public Properties ------ 
    @property
    def PA(self):
        """
        Pressure Amplitude [bar]
        """
        return self._PA
    
    @PA.setter
    def PA(self, value):
        self._PA = value
        self.cp[9] = self._PA[0] * 1e5
        self.cp[10]= self._PA[1] * 1e5

    @property
    def R0(self):
        """
        Equilibrium Radius [micron]
        """
        return self._R0
    
    @R0.setter
    def R0(self, value):
        self._R0 = value
        self._update_control_parameters()

    @property
    def FREQ(self):
        """
        Excitation Frequencies [kHz]
        """
        return self._FREQ
    
    @FREQ.setter
    def FREQ(self, value):
        self._FREQ = value
        self._update_control_parameters()


    # -------- Constructor -----------

    def __init__(self,
                 R0: float,
                 FREQ: List[float],
                 PA: List[float],
                 REL_FREQ: Optional[float] = None,
                 AC_FIELD: str = "CONST") -> None:
        """

        SingleBubble Object \n
            Simulate 1D coupled translational and radial bubble motion in dual-frequency acoustic field

        
        Arguments: \n
            R0   :   float        - equilibrium bubble radius [micron] \n
            FREQ : [float, float] - excitation frequency [kHz]   (dual frequency) \n
            PA   : [float, float] - pressure amplitude [bar] \n
            REL_FREQ: float       - relative freuqnecy (specify time-scale) [kHz]\n
            AC_FIELD : str (CONST, SW) - Acoustic field type \n
                            - CONST --> p(x, t) = PA0 * sin(w0 t) + PA1 * sin(w1 t + phi)  (homogeneos pressure field) \n
                            - SW    --> p(x, t) = PA0 * sin(k0 x) * sin(w0 t) + PA1 * sin(k1 x) * sin(w1 t) (standing acoustic wave) \n

        Example: \n
            model = SingleBubble(R0=40.0, FREQ=[25.0, 50.0], PA=[0.3, 0.0], AC_FIELD="CONST")
            model.integrate()
        """
        setup(AC_FIELD)
        self._REF_FREQ = REL_FREQ if REL_FREQ is not None else FREQ[0]
        self._FREQ = FREQ
        self._PA   = PA
        self._R0   = R0

        # Initial Conditions
        self.r0 = 1.0       # Bubble Radius
        self.u0 = 0.0       # Bubble Wall Veloctiy
        self.x0 = 0.0       # Bubble Position
        self.v0 = 0.0       # Bubble Velocity

        # Timedomain
        self.t0 = 0.0
        self.T  = 100       # Number Of Periods

        # ----- 
        self.cp = np.zeros((27, ), dtype=np.float64)
        self._update_control_parameters()


    def integrate(self, 
                  atol:float = 1e-9,
                  rtol: float = 1e-9,
                  min_step:float = 1e-16,
                  **options):
        
        """
        Returns:

        t (np.float64[:]) - dimensionless time \n
        r (np.float64[:]) - dimensionless radius \n
        x (np.float64[:]) - dimensionless position \n
        u (np.float64[:]) - dimensionless wall velocity \n 
        v (np.float64[:]) - dimensionless velocity \n
        """
        
        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.t0+self.T],
                        y0=[self.r0, self.x0, self.u0, self.v0],
                        args= (self.cp, ),
                        method='LSODA',
                        atol = atol,
                        rtol = rtol,
                        min_step = min_step,
                        )

        return res.t, res.y[0], res.y[1], res.y[2], res.y[3]



    def _update_control_parameters(self):
        """
        Calculates the control parameters
        """
        # Wave Length 
        l1 = CL / (self._FREQ[0] * 1000)
        l2 = CL / (self._FREQ[1] * 1000)
        lr = CL / (self._REF_FREQ * 1000)

        # Angular Frequency
        w1 = 2.0 * np.pi * (self._FREQ[0] * 1000)
        w2 = 2.0 * np.pi * (self._FREQ[1] * 1000)
        wr = 2.0 * np.pi * (self._REF_FREQ * 1000)

        # Convert to SI units
        R0      = self._R0 * 1e-6        # Equilibrum bubble radius (m)
        Pinf    = P0 * 1e5               # Ambient pressure (Pa) 

        # Parameters of the Keller--Miksis equation
        self.cp[0] = (2.0 * ST / R0 + Pinf - PV) * (2.0* np.pi / R0 / wr)**2.0 / RHO
        self.cp[1] = (1.0 - 3.0*PE) * (2 * ST / R0 + Pinf - PV) * (2.0*np.pi / R0 / wr) / CL / RHO
        self.cp[2] = (Pinf - PV) * (2.0 *np.pi / R0 / wr)**2.0 / RHO
        self.cp[3] = (2.0 * ST / R0 / RHO) * (2.0 * np.pi / R0 / wr)**2.0
        self.cp[4] = 4.0 * VIS / RHO / (R0**2.0) * (2.0* np.pi / wr)
        self.cp[5] = ((2.0 * np.pi / R0 / wr)**2.0) / RHO
        self.cp[6] = ((2.0 * np.pi / wr)** 2.0) / CL / RHO / R0
        self.cp[7] = R0 * w1 / (2 * np.pi) / CL
        self.cp[8] = 3.0 * PE

        # Physical Parameters
        self.cp[9]  = self._PA[0] * 1e5
        self.cp[10] = self._PA[1] * 1e5
        self.cp[11] = w1
        self.cp[12] = w2
        self.cp[13] = 0.0                     # Phase shift
        self.cp[14] = R0

        # Parameters for translation
        self.cp[15] = (l1 / R0)**2
        self.cp[16] = (2.0 * np.pi) / RHO / R0 / l1 / (wr * R0)**2.0
        self.cp[17] = 4 * np.pi / 3.0 * R0**3.0
        self.cp[18] = 12 * np.pi * VIS * R0

        # Acoustic field properties
        self.cp[19] = 2 * np.pi / l1        # k1 wavenumber
        self.cp[20] = 2 * np.pi / l2        # k2 wavenumber
        self.cp[21] = 1.0/ l1               # wavelength
        self.cp[22] = 1.0/ l2               # wavelength
        self.cp[23] = 1.0 / RHO / CL
        self.cp[24] = CL                    # Reference velocity
        self.cp[25] = 1.0 / wr              # Reference frequency
        self.cp[26] = lr                    # Reference length

    
if __name__ == "__main__":
    R0   = 30.0
    PA   = [0.7, 0.0]
    FREQ = [25.0, 50.0]
    model = SingleBubble(R0, FREQ, PA, AC_FIELD="SW")
    model.x0 = 0.123
    model.T = 5000
    t, _, x, _, _ = model.integrate()
    plt.plot(t, x, "k-")
    plt.show()

    
