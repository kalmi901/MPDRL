"""
This is a single-file implementation fo the model `KM1D2B`


"""

import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
import pandas as pd

# MATERIAL PROPERTIES (SI UNITS)
PV  = 0.0       # Vapour Pressure [Pa]
RHO = 998.0     # Liquod Density [kg/m**3]
ST  = 0.0725    # Surface Tension [N/m]
VIS = 0.001     # Liquid viscosity [Pa s]
CL  = 1500      # Liqid Sound Speed
P0  = 1.013*1e5 # Ambient Pressure [Pa]
PE  = 1.4       # Polytrophic Exponent [-]


# Try not to modify
__AC_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:], nb.float64[:])
__ODE_FUN_SIG = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:,::1], nb.float64[:], nb.float64[:])
__COL_THRESHOLD = 0.25


def setup(ac_field, k):
    global _PA, _PAT, _GRADP, _UAC
    global _ode_function
    global _collision_event
    
    # ----------------- Acoustic Field --------------------
    if ac_field == "CONST":
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PA(t, x, sp, dp):
            """
            Excitation Pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless bubble positions
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants
            
            Returns: \n
                p (nb.float64[:])   - Pressure amplitude
            """

            p = np.zeros_like(x)
            for i in range(k):
                p += dp[i] * np.sin(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return p

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, sp, dp):
            """
            The time derivative of the excitation pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless postion of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                pt(nb.float64[:])    - time derivative of pressure amplitude 
            """

            pt = np.zeros_like(x)
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                           * np.cos(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])
                
            return pt

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, sp, dp):
            """
            The gradient of the pressure field \n
            Argumens: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                px (nb.float64[:])  - Gradient of the pressure field (dp/dx)
                px = 0  for homogeneous pressure field

            """
            
            return np.zeros_like(x)
        
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, sp, dp):
            """
            The particle velocity induced by the acoustic irradiation
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimnesionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.gloat64[:])   - Dynamic constants

            Returns: \n
                ux (nb.float64)      - Particle velocity ux
                ux = 0  for homogeneous pressure field
            """

            return np.zeros_like(x)

    elif ac_field == "SW_A":
        """Standing Wave with ANTINODE located at x = 0"""
        
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PA(t, x, sp, dp):
            """
            Excitation Pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless position of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                p (nb.float64[:])   - Pressure amplitude
            """

            p = np.zeros_like(x)
            for i in range(k):
                p += dp[i]  * np.cos(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                            * np.sin(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return p

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PAT(t, x, sp, dp):
            """
            The time derivative of the excitation pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless postion of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                pt(nb.float64[:])    - time derivative of pressure amplitude
            """

            pt = np.zeros_like(x)
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                           * np.cos(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * np.cos(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return pt

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _GRADP(t, x, sp, dp):
            """
            The gradient of the pressure field \n
            Argumens: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                px (nb.float64[:])  - Gradient of the pressure field (dp/dx)
            """

            px = np.zeros_like(x)
            for i in range(k):
                px-= dp[i] * dp[3*k + i] \
                           * np.sin(2*np.pi*sp[2]*dp[3*k + i] * x + dp[2*k + i]) \
                           * np.sin(2*np.pi*sp[1]*dp[  k + i] * t + dp[2*k + i]) 
            
            return px

        @nb.njit(__AC_FUN_SIG, inline='always')
        def _UAC(t, x, sp, dp):
            """
            The particle velocity induced by the acoustic irradiation
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimnesionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.gloat64[:])   - Dynamic constants

            Returns: \n
                ux (nb.float64)      - Particle velocity ux
            """

            ux = np.zeros_like(x)
            for i in range(k):
                ux-= dp[i] * sp[4] \
                           * np.sin(2*np.pi*sp[2]*dp[3*k +i] * x + dp[2*k + i]) \
                           * np.cos(2*np.pi*sp[1]*dp[  k +i] * t + dp[2*k + i])

            return ux
        
    elif ac_field == "SW_N":
        """Standing Wave with NODE located at x = 0"""
        
        @nb.njit(__AC_FUN_SIG, inline='always')
        def _PA(t, x, sp, dp):
            """
            Excitation Pressure \n
            Arguments: \n
                t (nb.float64)      - Dimensionless time
                x (nb.float64[:])   - Dimensionless position of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                p (nb.float64[:])   - Pressure amplitude
            """

            p = np.zeros_like(x)
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
                x (nb.float64[:])   - Dimensionless postion of the bubble
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                pt(nb.float64[:])    - time derivative of pressure amplitude
            """

            pt = np.zeros_like(x)
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
                x (nb.float64[:])   - Dimensionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.float64[:])   - Dynamic constants

            Returns: \n
                px (nb.float64[:])  - Gradient of the pressure field (dp/dx)
            """

            px = np.zeros_like(x)
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
                x (nb.float64[:])   - Dimnesionless bubble position
                sp(nb.float64[:])   - Static constants
                dp(nb.gloat64[:])   - Dynamic constants

            Returns: \n
                ux (nb.float64)      - Particle velocity ux
            """

            ux = np.zeros_like(x)
            for i in range(k):
                ux+= dp[i] * sp[4] \
                           * np.cos(2*np.pi*sp[2]*dp[3*k +i] * x + dp[2*k + i]) \
                           * np.cos(2*np.pi*sp[1]*dp[  k +i] * t + dp[2*k + i])

            return ux

    # ------------------ ODE Function ---------------------

    @nb.njit(__ODE_FUN_SIG)
    def _ode_function(t, x, cp, sp, dp):
        """
        Dimensionless Keller--Miksis equation for pair-bubbles coupled wiht 1 dimensional translational motion \n
        for details see the model description of KM1D2B \n
        __________ \n
        Arguments: \n
            t (nb.float64)              - Dimensionless time
            x (nb.float64[:])           - State variables (dimless R0, R1, z0, z1, U0, U1, w0, w1)
            cp(nb.float64[:])           - Pre-computed constants
            sp(nb.float64[:])           - Static (bubble size independent) constants
            dp(nb.float64[:])           - Dynamic constants

        Returns: \n
            dx(nb.float64[:])           dx/dt = A(t)^-1 f(x,t) \n \n

        
        """

        x0i = x[0:2]
        x1i = x[2:4]
        x2i = x[4:6]
        x3i = x[6:8]

        # Reversed variables
        x0j = np.flip(x0i)
        x2j = np.flip(x2i)
        x3j = np.flip(x3i)
        s = np.array((1, -1), dtype=np.float64)              
        rd = 1.0 / abs(x1i[0]-x1i[1])                                               # Dimensionless distance 

        dx = np.zeros_like(x)
        rx0 = 1.0 / x0i
        p = rx0**sp[0]

        N = (cp[0] + cp[1]*x2i) * p \
                - cp[2] * (1 + cp[7]*x2i) - cp[3]* rx0 - cp[4]*x2i*rx0 \
                - 1.5 * (1.0 - cp[7]*x2i * (1.0/3.0))*x2i*x2i \
                - (1 + cp[7]*x2i) * cp[5] * _PA(t, x1i, sp, dp) - cp[6] * _PAT(t, x1i, sp, dp) * x0i \
                + cp[8] * x3i*x3i \
                + cp[13] * rd * (-2*x0j*x2j**2                                      
                            + 0.5 * rd*x0j**2 
                            * ( s*(x3i*x2j + x2j*x3j) 
                            - rd*x0j*x3j*(x3i + 2*x3j) ))

        D = x0i - cp[7]*x0i*x2i + cp[4]*cp[7]
        rD = 1.0 / D

        vj = cp[12] * x0j*x0j * rd*rd * (-s*x2j + x0j*x3j*rd)

        Fb1 = - cp[10]*x0i*x0i*x0i * _GRADP(t, x1i, sp, dp)                         # Primary Bjerknes Force
        Fd  = - cp[11]*x0i * (x3i*sp[3] - _UAC(t, x1i, sp, dp) - vj)                # Drag Force

        du = 3*(Fb1+Fd)*cp[9]*rx0*rx0*rx0 - 3.0*x2i*rx0*x3i \
            + cp[14] *rd*rd*x0j * (-s*x2j * (x0j*x2i + 2*x0i*x2j)
                                   +x0j*rd*x3j * (x0j*x2i + 5*x0i*x2j) ) * rx0


        A = np.eye(4)
        tmp = cp[13] * x0j*x0j * rd * rD
        A[0, 1], A[1, 0] = tmp
        A[0, 3], A[1, 2] =-tmp * 0.5*s * x0j * rd
        tmp = cp[14] * x0i * x0j*x0j * rd*rd * rx0
        A[2, 1], A[3, 0] = tmp * s
        A[2, 3], A[3, 2] = -tmp * x0j*rd

        dx[0:2] = x2i
        dx[2:4] = x3i
        dx[4:8] = np.linalg.solve(A, np.hstack((N * rD, du )))

        return dx
    

    # ----- Event Bubble Collision ------
    @nb.njit
    def _collision_event(t, x, cp, sp, dp): 
        return abs(x[3]-x[2]) - np.dot(x[0:2], cp[15]) * (1 + __COL_THRESHOLD)
    _collision_event.terminal = True
    _collision_event.direction = 0 


class DualBubble:

    def __init__(self,
                 R0: List[float],
                 FREQ: List[float],
                 PA: List[float],
                 PS: List[float]=[0.0, 0.0],
                 REF_FREQ: Optional[float] = None,
                 k: int = 1,
                 AC_FIELD: str = "CONST") -> None:
        """
        Arguments: \
            R0          - equilibrium bubble radii [R00, R01] (micron)
            FREQ        - excitation frequency [f0... fk-1] (kHz)
            PA          - pressure amplitude [PA0... PAk-1] (bar)
            PS          - phase shift [PS0... PSk-1] (radians)
            REF_FREQ    - relative frequency (optional, default FREQ[0]) (kHz)
            k           - number of harmonic components (int)
            AC_FIELD    - acoustic field type
                        - CONST (Homogeneous pressure field)
                        - SW_A (Standing wave with central antionde
                        - SW_N (Standint wave with central node)

        """

        setup(AC_FIELD, k)  # Call Swtup to initializ the model equations
        # Convert to SI Units
        self._R0 = np.array(R0, dtype=np.float64) * 1e-6
        self._FREQ = np.array(FREQ, dtype=np.float64) * 1e3
        self._PA = np.array(PA, dtype=np.float64) *1e5

        self._PS = np.array(PS, dtype=np.float64)
        self._REF_FREQ = REF_FREQ*1e3 if REF_FREQ is not None else FREQ[0]*1e3
        self._k = k

        # Initial Consitions
        self.r0 = np.ones(( 2, ), dtype=np.float64)
        self.u0 = np.zeros((2, ), dtype=np.float64)
        self.x0 = np.zeros((2, ), dtype=np.float64)
        self.v0 = np.zeros((2, ), dtype=np.float64)

        # TimeDomain
        self.t0 = 0.0
        self.T = 100.0      # Number of Periods
        self.cp = np.zeros((16, 2), dtype=np.float64, order='C')
        self.sp = np.zeros(( 5,  ), dtype=np.float64)
        self.dp = np.zeros((4*k, ), dtype=np.float64)

        self._update_control_parameters()


    def integrate(self,
                atol: float = 1e-9,
                rtol: float = 1e-9,
                min_step: float = 1e-16,
                event: Optional[Callable]=None,
                **options):
        
        """
        Returns: \
            t (np.float64[:])   - dimensionless time
            r (np.float64[:])   - dimensionless bubble radii
            u (np.float64[:])   - dimensionless bubble wall velocities
            x (np.float64[:])   - dimensionless bubble positions
            v (np.float64[:])   - dimensionless bubble translational velocities
        """

        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.T],
                        y0=np.hstack((self.r0, self.x0, self.u0, self.v0)),
                        args=(self.cp, self.sp, self.dp),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step,
                        events=_collision_event)

        # TODO: **options are not used
        
        return res.t, res.y[0:2], res.y[4:6], res.y[2:4], res.y[6:8]

    def _update_control_parameters(self):
        """"
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

        # Coupling Constants
        self.cp[12] = (np.flip(self._R0) / lr)**3 * CL
        self.cp[13] = np.flip(self._R0)**3 / self._R0**2 / lr
        self.cp[14] = 3*(np.flip(self._R0) / lr)**3
        self.cp[15] = self._R0 / lr

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
    R0 = [6.0, 5.0]       # (micron)
    FREQ = [20.0, 25.0]     # (kHz)
    PA = [-1.20 * P0*1e-5, 0.5]       # (bar)

    lr = CL / (FREQ[0] * 1000)
    
    km_bubbles = DualBubble(R0, FREQ, PA, k=1, AC_FIELD="CONST")

    km_bubbles.r0[0] = 1
    km_bubbles.r0[1] = 1
    km_bubbles.x0[1] = 300 * 1e-6 / lr
    km_bubbles.T = 10

    t, r, _, x, _ = km_bubbles.integrate()


    #pd.DataFrame(np.hstack((t[:, np.newaxis], r[0][:, np.newaxis], r[1][:, np.newaxis],
    #                       x[0][:, np.newaxis], x[1][:, np.newaxis])),
    #                       dtype=np.float64).to_csv("KM2B_SOL_DEBUG.csv", header=False, index=False)

    plt.figure(1)
    plt.plot(t, r[0], "k-")
    plt.plot(t, r[1], "r--")

    plt.figure(2)
    plt.plot(t, (x[1]-x[0]) * lr * 1e6, "r-", label="D(t)")
    plt.plot(t, (r[0]*R0[0] + r[1]*R0[1]), "b-", label="$R_0(t)+R_1(t)$")

    plt.show()

