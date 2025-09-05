"""
Single-file implementation of the KM1DNBC model 
(Keller–Miksis 1D N-Bubble Coupled System)

This module implements a simplified, CPU-based version of the KM1DNBC system to support testing and validation of the main GPU-based solver.

Key features:
- Models a chain of N globally coupled bubbles governed by the Keller–Miksis equation.
- Intended for theoretical validation and debugging — not optimized for performance.
- Matches should be verified against limiting cases (e.g., KM1D2B for 2 bubbles).
- Matrix decomposition is applied to mimic the logic of the GPU code

NOTE: This file intentionally contains copy-pasted code from previous implementations to keep it self-contained and to preserve logic parallelism with GPU implementations. 
The goal here is clarity and traceability — not software modularity.
"""


import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
import time

# MATERIAL PROPERTIES (SI UNITS)
PV  = 0.0       # Vapour Pressure [Pa]
RHO = 998.0     # Liquod Density [kg/m**3]
ST  = 0.0725    # Surface Tension [N/m]
VIS = 0.001     # Liquid viscosity [Pa s]
CL  = 1500      # Liqid Sound Speed
P0  = 1.013*1e5 # Ambient Pressure [Pa]
PE  = 1.4       # Polytrophic Exponent [-]

# Try not to modify 
__AC_FUN_SIG    = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:], nb.float64[:])
__ODE_FUN_SIG   = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:,::1], nb.float64[:], nb.float64[:], nb.float64[:,:,:], nb.float64, nb.float64, nb.int64)
__EVENT_FUN_SIG = nb.float64(nb.float64, nb.float64[:], nb.float64[:,::1], nb.float64[:], nb.float64[:], nb.float64[:,:,:], nb.float64, nb.float64, nb.int64)
__COL_THRESHOLD = 0.25          # Collision Threshold between bubbles


def setup(ac_field, k, num_bubbles, linsolve: str = "bicg"):
    global _PA, _PAT, _GRADP, _UAC
    global _ode_function
    global _collision_event
    global _lsolve

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
        
    # ------------- ODE SYSTEMS -------------

    # - Implicit Couplings / Linalg Solvers -

    @nb.njit(inline='always', cache=True)
    def _AxV(r_delta, s, mx, G, H, v):
        """
        A00 = G[0] * (mx[0] * r_delta) @ diag(H[5]) + I
        A01 = 0.5 * G[0] * (mx[0] * r_delta**2 * s) @ diag(H[6])
        A10 = -(mx[1] * r_delta**2 * s) @ diag(H[5])
        A11 = -(mx[1] * r_delta**3) @ diag(H[6]) + I
        """
        x = np.zeros_like(v)
        C00 = mx[0] * r_delta
        x[:num_bubbles] += G[0] * (C00 @ (H[5] * v[:num_bubbles])) + v[:num_bubbles]

        C01 = C00 * r_delta * s
        x[:num_bubbles] += 0.5 * G[0] * (C01 @ (H[6] * v[num_bubbles:]))

        C10 = mx[1] * r_delta**2
        x[num_bubbles:] += -(C10 * s) @ (H[5] * v[:num_bubbles])

        C11 = C10 * r_delta
        x[num_bubbles:] += -C11 @ (H[6] * v[num_bubbles:]) + v[num_bubbles:]

        return x

    @nb.njit(inline='always', cache=True)
    def _ATxV(r_delta, s, mx, G, H, v1, v2):
        """
        A00 = G[0] * (mx[0] * r_delta) @ diag(H[5]) + I
        A01 = 0.5 * G[0] * (mx[0] * r_delta**2 * s) @ diag(H[6])
        A10 = -(mx[1] * r_delta**2 * s) @ diag(H[5])
        A11 = -(mx[1] * r_delta**3) @ diag(H[6]) + I
        """
        x1 = np.zeros_like(v1)      # AxV
        x2 = np.zeros_like(v2)      # ATxV
        
        C00 = mx[0] * r_delta
        x1[:num_bubbles] += G[0] * (C00 @ (H[5] * v1[:num_bubbles]))   + v1[:num_bubbles]
        x2[:num_bubbles] += H[5] * (C00.T @ (G[0] * v2[:num_bubbles])) + v2[:num_bubbles]

        C01 = C00 * r_delta * s
        x1[:num_bubbles] += 0.5 * G[0] * (C01 @ (H[6] * v1[num_bubbles:]))
        x2[num_bubbles:] += 0.5 * H[6] * (C01.T @ (G[0] * v2[:num_bubbles]))

        C10 = mx[1] * r_delta**2
        x1[num_bubbles:] += -(C10 * s) @ (H[5] * v1[:num_bubbles])
        x2[:num_bubbles] += -H[5] * ((C10 * s).T @ v2[num_bubbles:])

        C11 = C10 * r_delta
        x1[num_bubbles:] += -C11 @ (H[6] * v1[num_bubbles:]) + v1[num_bubbles:]
        x2[num_bubbles:] += -H[6] * (C11.T @ v2[num_bubbles:]) + v2[num_bubbles:]

        return x1, x2


    @nb.njit()
    def _bicg(r_delta, s, mx, G, H, b, atol=1e-10, rtol=1e-10, maxiter=100):
        x = b.copy()
        r = b - _AxV(r_delta, s, mx, G, H, x)
        rt = r.copy()  # shadow residual
        bnorm = np.linalg.norm(b)
        tol = max(atol, rtol * bnorm)
        rnorm = np.linalg.norm(r)
        if rnorm <= tol:
            return x, 0  # már konvergált

        eps = 1.0e-30
        rho_old = 1.0  # dummy init; az első iterációban nem használjuk

        # p, pt az első körben r, rt
        p = r.copy()
        pt = rt.copy()

        for k in range(maxiter):
            # breakdown ellenőrzés
            rho = r @ rt
            if np.abs(rho) < eps:
                return x, -3  # BiCG breakdown (rho ~ 0)

            if k > 0:
                beta = rho / rho_old
                p = r + beta * p
                pt = rt + beta * pt

            #q = A @ p
            #qt = A.T @ pt  # <-- javítás: pt, nem p!
            q, qt = _ATxV(r_delta, s, mx, G, H, p, pt)

            d = pt @ q
            if np.abs(d) < eps:
                return x, -2  # BiCG breakdown (d ~ 0)

            alpha = rho / d
            x = x + alpha * p
            r = r - alpha * q
            rt = rt - alpha * qt

            rnorm = np.linalg.norm(r)
            if rnorm <= tol:
                return x, 1  # konvergált

            rho_old = rho  # <-- javítás: a végén frissítünk

        return x, -1  # maxiter

    @nb.njit()
    def _bicgstab(r_delta, s, mx, G, H, b, atol=1e-10, rtol=1e-10, maxiter=100):
        """
        Warm start: x = b.copy()  (initilized wiht uncoupled rhs).
        Visszatérés: (x, status)
        1  = Converged
        0  = Converged on initialization
        -1  = max iter reacehed
        -2  = breakdown: rhat·v ~ 0
        -3  = breakdown: rho ~ 0
        -4  = breakdown: t·t ~ 0
        -5  = breakdown: omega ~ 0
        """
        # Warm start a csatolásmentes megoldással
        x = b.copy()
        r = b - _AxV(r_delta, s, mx, G, H, x)
        rhat = r.copy()  # árnyék reziduum

        bnorm = np.linalg.norm(b)
        tol = max(atol, rtol * bnorm)
        rnorm = np.linalg.norm(r)
        if rnorm <= tol:
            return x, 0

        eps = 1.0e-300
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0

        v = np.zeros_like(b)
        p = np.zeros_like(b)

        for k in range(maxiter):
            rho = rhat @ r
            if np.abs(rho) < eps:
                return x, -3  # breakdown (rho ~ 0)

            if k == 0:
                p = r.copy()
            else:
                beta = (rho / rho_old) * (alpha / omega)
                p = r + beta * (p - omega * v)

            #v = A @ p
            v = _AxV(r_delta, s, mx, G, H, p)
            denom = rhat @ v
            if np.abs(denom) < eps:
                return x, -2  # breakdown (rhat·v ~ 0)

            alpha = rho / denom
            ss = r - alpha * v

            # Fél-lépés ellenőrzés
            snorm = np.linalg.norm(ss)
            if snorm <= tol:
                x = x + alpha * p
                return x, 1

            #t = A @ s
            t = _AxV(r_delta, s, mx, G, H, ss)
            tt = t @ t
            if np.abs(tt) < eps:
                return x, -4  # breakdown (t·t ~ 0)

            omega = (t @ ss) / tt

            x = x + alpha * p + omega * ss
            r = ss - omega * t

            rnorm = np.linalg.norm(r)
            if rnorm <= tol:
                return x, 1

            if np.abs(omega) < eps:
                return x, -5  # breakdown (omega ~ 0)

            rho_old = rho

        return x, -1

    _lsolve = _bicg if linsolve == "bicg" else _bicgstab

    # ----------- Explicit Coupling -------------
    @nb.njit()
    def _explicit_coupling(r_delta, s, mx, G, H):
        """
        Calculate the explicit coupling \n
        ______________\n
        Arguments: \n
            r_delta (nb.float64[:,:])    - Inverse of the distance matrix
            s       (nb.float64[:,:])    - Direction sign matrix
            mx      (nb.float64[:,:,:])  - Constant coupling matrix
            G       (nb.float64[:])      - Coupling Factors
            H       (nb.float64[:])      - Coupling Terms
        Retruns \n
            dx_c    (nb.float64[:])      - Correction of the RHS
        """

        dx_c = np.zeros((2*num_bubbles, ), dtype=np.float64)
        # --- Radial Couplings ---
        # First-order
        coupling_matrix = mx[0] * r_delta
        dx_c[: num_bubbles] +=  -2.0 * (coupling_matrix @ H [0] ) * G[0]

        # Second-order
        coupling_matrix = coupling_matrix * r_delta
        dx_c[: num_bubbles] +=  -0.5 * ((coupling_matrix * s) @ H[1]) * G[1] \
                                -2.5 * ((coupling_matrix * s) @ H[2]) * G[0]
        
        # Third-order
        coupling_matrix = coupling_matrix * r_delta
        dx_c[: num_bubbles] +=  -0.5 * (coupling_matrix @ H[3]) * G[1] \
                                -1.0 * (coupling_matrix @ H[4]) * G[0]

        # --- Translational Couplings ---
        # Second-Order
        coupling_matrix = mx[1] * r_delta**2
        dx_c[num_bubbles :] +=  2.0 * ((coupling_matrix * s) @ H[0]) \
                                    + ((coupling_matrix * s) @ H[1]) * G[2]
        
        # Third-order
        coupling_matrix = coupling_matrix * r_delta
        dx_c[num_bubbles :] +=  5.0 * (coupling_matrix @ (H[2])) \
                                    + (coupling_matrix @ (H[3])) * G[2]

        # --- Translational / Liquid Velocity ---
        # Second order
        coupling_matrix = mx[2] * r_delta**2
        dx_c[num_bubbles :] += ((coupling_matrix * s) @ H[1]) * G[3]
        # Third order
        coupling_matrix = coupling_matrix * r_delta
        dx_c[num_bubbles :] += (coupling_matrix @ H[3]) * G[3]

        return dx_c

    # ---------- Main ODE Function --------------
    nb.njit(__ODE_FUN_SIG)
    def _ode_function(t, x, up, gp, dp, mx, latol, lrtol, maxiter):
        """
        Dimensionless Keller--Miksis equation for N-bubble coupled system \n
        ____________\n
        Arguments: \n
            t (nb.float64)      - Dimensionless time
            x (nb.float64[:])   - State variables
            up (nb.float64[:])  - Unit pre-computed constants

        """

        # ---- State Variables ----    
        x0 = x[            0:   num_bubbles]                        # Radius
        x1 = x[  num_bubbles: 2*num_bubbles]                        # Position
        x2 = x[2*num_bubbles: 3*num_bubbles]                        # Wall Velocity
        x3 = x[3*num_bubbles: 4*num_bubbles]                        # Wall Acceleration

        dx = np.zeros_like(x)

        # ---- Uncoupled Part ----
        rx0 = 1.0 / x0
        p = rx0**gp[0]

        N = (up[0] + up[1]*x2) * p \
                - up[2] * (1 + up[7]*x2) - up[3]* rx0 - up[4]*x2*rx0 \
                - 1.5 * (1.0 - up[7]*x2 * (1.0/3.0))*x2*x2 \
                - (1 + up[7]*x2) * up[5] * _PA(t, x1, gp, dp) - up[6] * _PAT(t, x1, gp, dp) * x0 \
                + up[8] * x3*x3 


        D = x0 - up[7]*x0*x2 + up[4]*up[7]
        rD = 1.0 / D

        Fb1 = - up[10]*x0*x0*x0 * _GRADP(t, x1, gp, dp)                         # Primary Bjerknes Force
        Fd  = - up[11]*x0 * (x3*gp[3] - _UAC(t, x1, gp, dp))                    # vij is inculdued as coupling term
        du = 3.0*((Fb1+Fd)*up[9]*rx0*rx0 - x2*x3) * rx0    
        
        dx[            0:   num_bubbles] = x2
        dx[  num_bubbles: 2*num_bubbles] = x3
        dx[2*num_bubbles: 3*num_bubbles] = N * rD
        dx[3*num_bubbles: 4*num_bubbles] = du


        # ---- COUPLING ----
        
        # Distance Between bubbles ----
        delta = x1[:, None] - x1[None, :]       # Delta with direction
        s     = np.sign(delta)                  # Direction Sign
        np.fill_diagonal(delta, 1.0)            # Avod division by zero...
        r_delta = 1.0 / np.abs(delta)           
        np.fill_diagonal(r_delta, 0.0)          # Set diagonal to zero

        # --- Coupling Factors ---
        G = np.zeros((4, num_bubbles), dtype=np.float64)
        G[0] = rD
        G[1] = x3 * rD
        G[2] = x2 * rx0
        G[3] = rx0**2

        # --- Coupling Terms ---
        H = np.zeros((7, num_bubbles), dtype=np.float64)
        H[0] = x0 * x2**2
        H[1] = x0**2 * x2
        H[2] = x0**2 * x2 * x3
        H[3] = x0**3 * x3
        H[4] = x0**3 * x3**2
        H[5] = x0**2
        H[6] = x0**3

        dx_c = _explicit_coupling(r_delta, s, mx, G, H)
        dx[2*num_bubbles:] += dx_c

        # --- Implicit Terms / Radial ---
        #A = np.zeros((2*num_bubbles, 2*num_bubbles), dtype=np.float64)
        #AT = np.zeros((2*num_bubbles, 2*num_bubbles), dtype=np.float64)
        # First order
        #coupling_matrix = mx[0] * r_delta
        #A[:num_bubbles, :num_bubbles] = rD[:, None] * (coupling_matrix * x0[None, :]**2)
        #AT[:num_bubbles, :num_bubbles] = x0[:,None]**2 * (coupling_matrix.T * rD[None, :])
        # Second order
        #coupling_matrix = coupling_matrix * r_delta * s
        #A[:num_bubbles, num_bubbles:] = 0.5 * rD[:, None] * (coupling_matrix * x0[None, :]**3)
        #AT[num_bubbles:, :num_bubbles] = 0.5 * x0[:, None]**3 * (coupling_matrix.T * rD[None, :])
        # --- Implicit Terms / Translational ---
        # Second order
        #coupling_matrix = mx[1] * r_delta**2
        #A[num_bubbles:, :num_bubbles] = -(coupling_matrix * s) * x0[None, :]**2
        #AT[:num_bubbles, num_bubbles:]= - x0[:, None]**2 * (coupling_matrix * s).T
        # Third order
        #coupling_matrix = coupling_matrix * r_delta
        #A[num_bubbles:, num_bubbles:] = -(coupling_matrix) * x0[None, :]**3
        #AT[num_bubbles:, num_bubbles:]= - x0[:, None]**3 * coupling_matrix.T

        # Fill Diagonal elements ...
        #np.fill_diagonal(A, 1.0)
        #np.fill_diagonal(AT, 1.0)
       
        sol, _ = _lsolve(r_delta, s, mx, G, H, dx[2*num_bubbles:4*num_bubbles], latol, lrtol, maxiter)
    
        dx[2*num_bubbles: 4*num_bubbles] = sol

        return dx

    @nb.njit(__EVENT_FUN_SIG)
    def _collision_event(t, x, up, _gp, _dp, _mx, _latol, _lrtol, _lmaxiter):
        x0 = x[:num_bubbles] * up[12]         # Radius rescaled
        x1 = x[  num_bubbles: 2*num_bubbles]
        #min_delta = (x0[:, None] + x0[None, :]) * (1 + __COL_THRESHOLD)     # add a small threshold
        #delta = np.abs(x1[:, None] - x1[None, :])
        #np.fill_diagonal(delta, np.Inf)
        # Simplfied event for chain of bubble, valid for increasing x coordinate
        min_delta = (x0[:-1] + x0[1:]) * (1 + __COL_THRESHOLD)
        delta = np.abs(x1[:-1] - x1[1:])
        return np.min(delta - min_delta)
    _collision_event.terminal = True
    _collision_event.direction = 0


class MultiBubble:
    def __init__(self,
                 R0: List[float],
                 FREQ: List[float],
                 PA: List[float],
                 PS: List[float],
                 REL_FREQ: Optional[float] = None,
                 k: int = 1,
                 AC_FIELD: str = "CONST",
                 LEN: float = 1.0,
                 LINSOLVE: str = "bicg") -> None:
        """
        Arguments
        R0          - equilibrium bubble radii [R00, R01, ... R0N-1] (micron)
        X0          - initial bubble coordinate [X0, X1, ... XN-1] (mm)
        FREQ        - excitation frequency [f0, f1, ... fk-1] (kHz)
        PA          - pressure amplitude [PA0, PA1, ... PAk-1] (bar)
        PS          - phase shift [PS0, PS1, ... PSk-1] (radians)
        REF_FREQ    - relative frequency (optional, default FREQ[0]) (kHz)
        k           - number of harmonic components (int)
        LINSOLVE    - linear solver for implicit coupling (bicg or bicgstab)
        """

        assert len(R0) >=2, "The number of bubbles must be larger than 2"
        setup(AC_FIELD, k, len(R0), LINSOLVE)          # Call Setup to initialize the model equations
        # CONVER TO SI UNTIS!
        self._R0    = np.array(R0, dtype=np.float64)   * 1e-6
        self._FREQ  = np.array(FREQ, dtype=np.float64) * 1e3
        self._PA    = np.array(PA, dtype=np.float64)   * 1e5
        self._PS    = np.array(PS, dtype=np.float64)
        self._REF_FREQ = REL_FREQ * 1e3 if REL_FREQ is not None else FREQ[0] * 1e3
        self._REF_L = CL / self._REF_FREQ           # Reference Wave length (m)

        self._k = k

        self.num_bubbles = len(R0)
        assert self.num_bubbles >=2, print(f"The minimum number of bubbles must be 2")
        
        # INITIAL CONDITIONS
        self.r0 = np.ones((self.num_bubbles,  ), dtype=np.float64)           # Dimensinless  Initial Radius
        self.u0 = np.zeros((self.num_bubbles, ), dtype=np.float64)           # Dimensionless Initial Wall velocity
        self.x0 = np.linspace(0.0, LEN, self.num_bubbles, endpoint=True)     # Dimenionless Position (mm-->m -->dimlesss)
        self.v0 = np.zeros((self.num_bubbles, ), dtype=np.float64)           # Dimensionless translational velocity

        # TIME DOMAIN 
        self.t0 = 0.0
        self.T  = 1.0


        # PLACEHOLDER FOR PRE-COMPUTED CONSTANTS
        self.up = np.zeros((13, self.num_bubbles), dtype=np.float64, order="C")                            # Unit Parameters  (register) 
        self.gp = np.zeros((5, ),   dtype=np.float64)                                       # Global Parameters (shared/globla)
        self.dp = np.zeros((4*k,),  dtype=np.float64)                                       # Dynamic Parameters (shared/global)
        self.mx = np.zeros((3, self.num_bubbles, self.num_bubbles), dtype=np.float64)       # Static Coupling Matrix (shared/global)
        
        self._update_control_parameters()


    def _update_control_parameters(self):
        """
        Pre-Computed constant parameters
        """

        wr = 2.0 * np.pi * self._REF_FREQ
        lr = CL / self._REF_FREQ

        # ----- UNIT PARAMETERS ------
        # ---- Keller--Miksis Eqn ----
        self.up[0] = (2.0 * ST / self._R0 + P0 - PV) * (2.0* np.pi / self._R0 / wr)**2.0 / RHO
        self.up[1] = (1.0 - 3.0*PE) * (2 * ST / self._R0 + P0 - PV) * (2.0*np.pi / self._R0 / wr) / CL / RHO
        self.up[2] = (P0 - PV) * (2.0 *np.pi / self._R0 / wr)**2.0 / RHO
        self.up[3] = (2.0 * ST / self._R0 / RHO) * (2.0 * np.pi / self._R0 / wr)**2.0
        self.up[4] = 4.0 * VIS / RHO / (self._R0**2.0) * (2.0* np.pi / wr)
        self.up[5] = ((2.0 * np.pi / self._R0 / wr)**2.0) / RHO
        self.up[6] = ((2.0 * np.pi / wr)** 2.0) / CL / RHO / self._R0
        self.up[7] = self._R0 * wr / (2 * np.pi) / CL

        # Translational-Motion Constansts
        self.up[8]  = (0.5 * lr / self._R0)**2
        self.up[9]  = (2.0 * np.pi) / RHO / self._R0 / lr / (wr * self._R0)**2.0
        self.up[10] = 4 * np.pi / 3 * self._R0**3
        self.up[11] = 12 * np.pi * VIS * self._R0
        self.up[12] = self._R0 / lr                                      # Collision Event

  
        # ---- COUPLING MATRIX  ----
        for i in range(self.num_bubbles):
            for j in range(self.num_bubbles):
                if i == j:
                    self.mx[:, i, j] = 0.0
                else:
                    # Radial
                    self.mx[0, i, j] = self._R0[j]**3 / self._R0[i]**2 / lr    # cp[13]
                    # Translational
                    self.mx[1, i, j] = 3.0 * (self._R0[j] / lr)**3             # cp[14]
                    # Liquid Velocity Terms
                    self.mx[2, i, j] = (18 * VIS / RHO) * (2*np.pi / wr) * (self._R0[j] / lr)**3 / (self._R0[i] **2)



        # ---- GLOBAL PARAMETERS -----
        self.gp[0] = 3 * PE
        self.gp[1] = 1.0 / wr
        self.gp[2] = lr / (2 * np.pi)
        self.gp[3] = CL
        self.gp[4] = 1.0 / RHO / CL

        # -- EXCITATION PARAMETERS ---
        for i in range(self._k):
            self.dp[          + i] = self._PA[i]                            # Pressure Amplitude
            self.dp[  self._k + i] = 2.0 * np.pi * self._FREQ[i]            # Angular Frequency
            self.dp[2*self._k + i] = self._PS[i]                            # Phase Shift
            self.dp[3*self._k + i] = 2.0 * np.pi * self._FREQ[i] / CL       # Wave number



    def integrate(self,
                    atol: float = 1e-9,
                    rtol: float = 1e-9,
                    latol: float = 1e-10,
                    lrtol: float = 1e-10,
                    maxiter: int = 100,
                    min_step:float = 1e-20,
                    event: Optional[Callable]=None,
                    **options):
        """
        Returns: \
            t (np.float64[:])   - dimensionless time
            r (np.float64[:])   - dimensionless bubble radius
        """

        res = solve_ivp(fun=_ode_function,
                        t_span=[self.t0, self.T],
                        y0=np.hstack((self.r0, self.x0, self.u0, self.v0)),
                        args=(self.up, self.gp, self.dp, self.mx, latol, lrtol, maxiter),
                        method='LSODA',
                        atol=atol,
                        rtol=rtol,
                        min_step=min_step,
                        events=_collision_event)
        
        return (res.t,
                res.y[                 0:1*self.num_bubbles],     # Radius
                res.y[2*self.num_bubbles:3*self.num_bubbles],     # Wall velocity
                res.y[1*self.num_bubbles:2*self.num_bubbles],     # Position
                res.y[3*self.num_bubbles:4*self.num_bubbles] )    # Translational


if __name__ == "__main__":

    # ------ ACOUSTIC FIELD PROPERTIES ------
    K = 1
    FREQ = [20.0, 40.0]               # (KHz)
    PA = [-1.20 * P0*1e-5, 0.0]       # (bar)
    PS   = [0.0, 0.0]
    LR   = CL / (FREQ[0] * 1000)      # Reference Lenght (m)

    # ----- INITIAL CONDITIONS -------------
    #R0 = [6.0, 5.0, 6.0, 5.0]*1        # Equilibrium Bubble Size (micron)
    R0 = [6.0, 5.0]        # Equilibrium Bubble Size (micron)

    multi_bubbles = MultiBubble(R0, FREQ, PA, PS, k=K, AC_FIELD="CONST", LEN=10000 * 1e-6 / LR,
                                LINSOLVE="bicgstab")

    multi_bubbles.T = 5
    multi_bubbles.r0[0] = 1.0
    multi_bubbles.r0[1] = 1.0
    multi_bubbles.u0[0] = 0.0
    multi_bubbles.u0[1] = 0.0
    multi_bubbles.v0[0] = 0.0
    multi_bubbles.v0[1] = 0.0
    multi_bubbles.x0[0] = 0 * 1e-6 / LR
    multi_bubbles.x0[1] = 300 * 1e-6 / LR
    #multi_bubbles.x0[2] = 10000 * 1e-6 / LR
    #multi_bubbles.x0[3] = 10300 * 1e-6 / LR


    start = time.time()
    t, r, _, x, _ = multi_bubbles.integrate()
    end = time.time()
    print(f"Number of Bubbles: {len(R0)}")
    print(f"Total simulation time: {end-start:.2f} seconds")


    plt.figure(1)
    plt.plot(t, (x[1]-x[0]) * LR * 1e6, "r-", label="D_{0-1}(t)")
    plt.plot(t, (r[0]*R0[0] + r[1]*R0[1]), "b-", label="$R_0(t)+R_1(t)$")

    #plt.figure(2)
    #plt.plot(t, (x[3]-x[2]) * LR * 1e6, "r-", label="D_{2-3}(t)")
    #plt.plot(t, (r[3]*R0[3] + r[2]*R0[2]), "b-", label="$R_2(t)+R_3(t)$")


    plt.figure(3)
    plt.plot(t, x[0] * LR * 1e6, 'k-')
    plt.plot(t, x[1] * LR * 1e6, 'r-')
    #plt.plot(t, x[2] * LR * 1e6, "b-")
    #plt.plot(t, x[3] * LR * 1e6, "g-")
    plt.ylabel(r"$x_i\, \mu m$")

    plt.show()




