"""
This is a single-file implementation of the model KM1D.

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
__AC_FUN_SIG = nb.float64(nb.float64, nb.float64, nb.float64[:], nb.float64[:])
__ODE_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])


def setup(ac_field, k):

    global _PA, _PAT, _GRADP, _UAC
    global _ode_function

    # ----------- Acoustic Field -------------
    if ac_field == "CONST":
        """
        Homogeneous pressure field (p_i(x,t) = P_{Ai} * sin(omega_i * t + phi_i) )
        """
        @nb.njit(__AC_FUN_SIG, inline="always")
        def _PA(t, x, cp):
            """
            Excitation Pressure (k-frequency) \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimensionless position of the bubble
                cp(nb.float64[:])   - Pre-computed constants

            Returns: \n
                p (nb.float64)      - Pressure amplitude
            """

            p = 0.0
            for i in range(k):
                p += cp[17 + i] * np.sin(2*np.pi*cp[9]*cp[17+k + i] * t + cp[17+2*k + i])

            return p

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, cp):
            """
            The time derivative of the excitation pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimensionless postion of the bubble
                cp(nb.float64[:])   - Pre-computed constants

            Returns: \n
                pt(nb.float64)       - time derivative of pressure amplitude
            """

            pt = 0.0
            for i in range(k):
                pt += cp[17 + i] * cp[16+k + i] \
                    * np.cos(2*np.pi*cp[9]*cp[17+k + i] * t + cp[17+2*k + i])

            return  pt

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, cp):
            """
            The gradient of the pressure field.
            Note: Zero for homogeneous pressure field
            """
            return 0.0

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, cp):
            """
            The particel velocity induced by the acoustic irradiation
            Note: Zero for homogeneous pressure field
            """
            return 0.0

    elif ac_field == "SW_A":
        """Standing Wawe with ANTINODE located at x = 0 """

        pass




    elif ac_field == "SW_N":
        """Standing Wave with NODE located at x = 0"""
        
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PA(t, x, sp, dp):
            """
            Excitation Pressure (dual-frequency) \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimensionless position of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                p (nb.float64)      - Pressure amplitude
            """

            p = 0.0
            for i in range(k):
                p += dp[i]  * np.sin(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                            * np.sin(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return p

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, sp, dp):
            """
            The time derivative of the excitation pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimensionless postion of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                pt(nb.float64)       - time derivative of pressure amplitude
            """

            pt = 0.0
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                           * np.sin(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * np.cos(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return pt

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, sp, dp):
            """
            The gradient of the pressure field \n
            Argumens: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimensionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                p (nb.float64)      - Gradient of the pressure field (dp/dx)
            """

            px = 0.0
            for i in range(k):
                px+= dp[i] * dp[3*k + i] \
                           * np.cos(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * np.sin(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i]) 
            
            return px

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, sp, dp):
            """
            The particle velocity induced by the acoustic irradiation
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64)      - Dimnesionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.gloat64[:])   - Dynamic constants

            Returns: \n
                ux (nb.float64)      - Particle velocity ux
            """

            ux = 0.0
            for i in range(k):
                ux+=-dp[i] * sp[4] \
                           * np.cos(2*np.pi*sp[2]*dp[3*k +i] * x + dp[2*k + i]) \
                           * np.cos(2*np.pi*sp[1]*dp[  k +i] * t + dp[2*k + i])

            return ux
            

    else:
        print(f"Error: Acoustic field type \"{ac_field}\" is not implemented!")
        exit()


    # ------------ ODE Function --------------

    @nb.njit(__ODE_FUN_SIG)
    def _ode_function(t, x, cp, sp, dp):
        """
        Dimensioness Keller--Miksis equation and 1 dimensional translational motion \n
        for details see the model description of KM1D \n
        _________ \n
        Arguments: \n
            t (nb.float64)          - Dimensionless time
            x (nb.float64[:])       - State variables (dimless R, z, U, w)
            cp(nb.float64[:])       - Pre-computed constants
            sp(nb.float64[:])       - Static constants
            dp(nb.float64[:])       - Dynamic constants

        Returns: \n
            dx(nb.float64[:])       - dx/dt = f(x, t)

            
        First-order ode System: \n
        
        x[0]                - Dimensionless Bubble Radius
        x[1]                - Dimensionless Bubble Position
        x[2]                - Dimensionless Bubble Wall Velocity
        x[3]                - Dimensionless Translational Velocity

        dx[0] = x[2]        - Dimensionless Bubble Wall Velocity
        dx[1] = x[3]        - Dimensionless Translational Velocity
        dx[2] = N/D         - Dimensionless Bubble Wall Acceleration
        dx[3] = Fex/Fref    - Dimensionless Translational Acceleration

        """

        dx = np.zeros_like(x)
        rx0 = 1.0 / x[0]
        p = rx0**sp[0]

        N = (cp[0] + cp[1]*x[2]) * p \
                - cp[2] * (1 + cp[7]*x[2]) - cp[3]* rx0 - cp[4]*x[2]*rx0 \
                - 1.5 * (1.0 - cp[7]*x[2] * (1.0/3.0))*x[2]*x[2] \
                - (1 + cp[7]*x[2]) * cp[5] * _PA(t, x[1], sp, dp) - cp[6] * _PAT(t, x[1], sp, dp) * x[0] \
                + cp[8] * x[3]*x[3]                                    # Feedback term

        D = x[0] - cp[7]*x[0]*x[2] + cp[4]*cp[7]
        rD = 1.0 / D

        Fb1 = - cp[10]*x[0]*x[0]*x[0] * _GRADP(t, x[1], sp, dp)           # Primary Bjerknes Force
        Fd  = - cp[11]*x[0] * (x[3]*sp[3] - _UAC(t, x[1], sp, dp) )       # Drag Force

        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = N * rD
        dx[3] = 3*(Fb1+Fd)*cp[9]*rx0*rx0*rx0 - 3.0*x[2]*rx0*x[3]

        return dx


class SingleBubble:

    def __init__(self, 
                 R0: float,
                 FREQ: List[float],
                 PA: List[float],
                 PS: List[float] = [0.0, 0.0],
                 REF_FREQ: Optional[float] = None,
                 k: int = 1,
                 AC_FIELD: str = "CONST") -> None:
        """
        Arguments: \n
            R0       - equilibrium bubble radius (micron)
            FREQ     - excitation frequecy [f0, f1] (kHz)
            PA       - pressure amplitude [PA0, PA1] (bar)
            PS       - phase shift [PS0, PS1] (radians)
            REL_FREQ - relative frequency (optional, default FREQ[0]) (kHz)
            k        - number of harmonic components (int)
            AC_FIELD - acoustic field type
                     - CONST (Homogeneous pressure field)
                     - SW_A  (Standing Wawe with central antinode)
                     - SW_N  (Standing Wave with central node)
        """

        setup(AC_FIELD, k)  # Call Setup to initialize the model equations
        # Convert to SI Units
        self._R0 = R0 * 1e-6
        self._FREQ = np.array(FREQ, dtype=np.float64) *  1e3
        self._PA = np.array(PA, dtype=np.float64) * 1e5
        self._PS = np.array(PS, dtype=np.float64)
        self._REF_FREQ = REF_FREQ*1e3 if REF_FREQ is not None else FREQ[0]*1e3
        self._k = k

        # Initial Conditions
        self.r0 = 1.0       # Bubble Radius
        self.u0 = 0.0       # Bubble Wall velcoity
        self.x0 = 0.0       # Bubble Position
        self.v0 = 0.0       # Bubble Velocity

        # Timedomain
        self.t0 = 0.0   
        self.T = 100.0      # Number of Periods
        self.cp = np.zeros((12, ), dtype=np.float64)
        self.sp = np.zeros(( 5, ), dtype=np.float64)
        self.dp = np.zeros((4*k,), dtype=np.float64)
        self._update_control_parameters()


    def integrate(self,
                atol: float = 1e-9,
                rtol: float = 1e-9,
                min_step: float = 1e-16,
                event: Optional[Callable] = None,
                **options):
        
        """
        Returns: \
            t (np.float64[:])   - dimensionless time
            r (np.float64[:])   - dimensionless bubble radius
            u (np.float64[:])   - dimensionless bubble wall velocity
            x (np.float64[:])   - dimensionless bubble position
            v (np.float64[:])   - dimensionless translational velocity
        """
        
        # TODO: **options are not used
        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.T],
                        y0=[self.r0, self.x0, self.u0, self.v0],
                        args=(self.cp, self.sp, self.dp),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step,
                        events=event)
        
        return res.t, res.y[0], res.y[2], res.y[1], res.y[3]
    

    def _update_control_parameters(self):
        """
        Pre-compute the constant parameters
        """

        wr = 2.0 * np.pi * self._REF_FREQ
        lr = CL / self._REF_FREQ

        # Keller--Miksis Constants
        self.cp[0] = (2.0 * ST / self._R0 + P0 - PV) * (2.0* np.pi / self._R0 / wr)**2.0 / RHO
        self.cp[1] = (1.0 - 3.0*PE) * (2 * ST / self._R0 + P0 - PV) * (2.0*np.pi / self._R0 / wr) / CL / RHO
        self.cp[2] = (P0 - PV) * (2.0 *np.pi / self._R0 / wr)**2.0 / RHO
        self.cp[3] = (2.0 * ST / self._R0 / RHO) * (2.0 * np.pi / self._R0 / wr)**2.0
        self.cp[4] = 4.0 * VIS / RHO / (self._R0**2.0) * (2.0* np.pi / wr)
        self.cp[5] = ((2.0 * np.pi / self._R0 / wr)**2.0) / RHO
        self.cp[6] = ((2.0 * np.pi / wr)** 2.0) / CL / RHO / self._R0
        self.cp[7] = self._R0 * wr / (2 * np.pi) / CL

        # Translational-Motion Constansts
        self.cp[8]  = (0.5 * lr / self._R0)**2
        self.cp[9]  = (2.0 * np.pi) / RHO / self._R0 / lr / (wr * self._R0)**2.0
        self.cp[10] = 4 * np.pi / 3 * self._R0**3
        self.cp[11] = 12 * np.pi * VIS * self._R0

        # Static Constants
        self.sp[0] = 3.0 * PE
        self.sp[1] = 1.0 / wr
        self.sp[2] = lr / (2 * np.pi) 
        self.sp[3] = CL
        self.sp[4] = 1 / RHO / CL


        # Excitation Parameters
        for i in range(self._k):
            self.dp[          + i] = self._PA[i]                            # Pressure Amplitude
            self.dp[  self._k + i] = 2.0 * np.pi * self._FREQ[i]            # Angular Frequency
            self.dp[2*self._k + i] = self._PS[i]                            # Phase Shift
            self.dp[3*self._k + i] = 2.0 * np.pi * self._FREQ[i] / CL       # Wave number



if __name__ == "__main__":
    R0 = 44.8               # (micron)
    FREQ = [31.25, 25.0]    # (kHz)
    PA   = [0.225, 0.5]     # (bar)

    km_bubble = SingleBubble(R0, FREQ, PA, k=1, AC_FIELD="SW_N")
    km_bubble.x0 = 0.25
    t, r, _, x, _ = km_bubble.integrate()

    plt.plot(t, r, "k-")
    #plt.plot(t, x, "k-")
    plt.show()