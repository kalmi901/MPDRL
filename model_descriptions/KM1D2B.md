# Model Description (KM1D2B)

- Keller--Miksis bubble model (KM)
- 1 dimensional translational motion (1D)
- 2 Bubbles (2B)
- Harmonic excitation up to $k$ components (Standing waves)

Note: The present model is the extension of model KM1D. 

### References

[1] J.B. Keller, M. Miksis, Bubble oscillations of large amplitude, J. Acoust. Soc. Am., 68(2), (1980), pp. 628-633. \
[2] A.A. Doinikov, Mathematical model for collective bubble dynamics in strong ultrasound fields, J. Acoust. Soc. Am., 116(2), (2004), pp. 821-827. \
[3]

## Keller--Miksis equation

The volumetric oscillation of the bubbles is described by the Keller--Miksis equation

```math
\left(1 - \dfrac{\dot{R}_i}{c_L} \right)R_i\ddot{R}_i + \left(1 -\dfrac{\dot{R}_i}{3c_L}\right) \dfrac{3}{2}\dot{R}_i^2 = \left(1 + \dfrac{\dot{R}_i}{c_L} + \dfrac{R_i}{c_L}\dfrac{d}{dt}\right) \dfrac{p_{Li}-p_{\infty}(z_i,t)}{\rho_L} + \dfrac{u_i^2}{4} + G_{ij}^{(rad)},
```

where $`R_i`$, and $`z_i`$ is the time-dependent radius, and position of the bubble $`i`$. The bubble index $`i\in(0, 1)`$ and $`j=1-i`$. The last term $`G_{ij}^{(rad)}`$ is the coupling term describing the effect of bubble $`j`$ on the radial oscillation of bubble $`i`$. The coupling terms are written as

```math
G_{ij}^{(rad)} = -\dfrac{R_j^2\ddot{R}_j + 2R_j\dot{R}_j^2}{D} + (-1)^i\dfrac{R_j^2\left(\dot{z}_i\dot{R}_j+R_j\ddot{z}_j +5\dot{R}_j \dot{z_j}\right)}{2D^2} - \dfrac{R_j^3\dot{z}_j\left(\dot{z}_i + 2 \dot{z}_j\right)}{2D^3} + \mathcal{O}\left(\dfrac{1}{D^4} \right)
```

where $`D = |z_j - z_i|`$ is the distance between the center of the bubbles.

The coupling term can be split into implicit and explicit ones $`G_{ij}^{(rad)} = G_{ij}^{(rad, impl)} + G_{ij}^{(rad, expl)}`$. Implicit terms contain second-order derivatives, explicit terms do not.

```math
G_{ij}^{(rad, impl)} = -\dfrac{R_j^2\ddot{R}_j}{D} + (-1)^i \dfrac{R_j^3\ddot{x}_j}{2D^2}
```

```math
G_{ij}^{(rad, expl)} = -\dfrac{2R_j\dot{R}_j^2}{D} + (-1)^i\dfrac{R_j^2\left(\dot{z}_i\dot{R}_j+5\dot{R}_j \dot{z_j}\right)}{2D^2} - \dfrac{R_j^3\dot{z}_j\left(\dot{z}_i + 2 \dot{z}_j\right)}{2D^3}
```

## Translational motion

The governing equation describing the translational bubble motion is

```math
R_i\ddot{z}_i+3\dot{R}_i\dot{z}_i=\dfrac{3F_{ex}(z_i,t)}{2\pi\rho_LR_i^2} + 3G_{ij}^{trn},
```

where $`G_{ij}^{(trn)}`$ is the coupling term that describes the effect of bubble $`j`$ on the translational motion of bubble $`i`$. This coupling term is written as

```math
G_{ij}^{(trn)}=-(-1)^i\dfrac{1}{D^2}\dfrac{d}{dt}\left(R_iR_j^2\dot{R}_j^2\right) + \dfrac{R_j^2\left(R_iR_j\ddot{x}_j+R_j\dot{R}_i\dot{x}_j+5R_i\dot{R}_j\dot{x}_j \right)}{D^3} + \mathcal{O}\left(\dfrac{1}{D^4} \right).
```

After calculating the time derivative, this coupling term can also be split into explicit and implicit terms:

```math
G_{ij}^{(trn,impl)}=-(-1)^i\dfrac{R_iR_j^2\ddot{R}_j}{D^2}+\dfrac{R_iR_j^3\ddot{x}_j}{D^3}
```

```math
G_{ij}^{(trn,expl)}=-(-1)^i\dfrac{\dot{R}_iR_j^2\dot{R}_j+2R_iR_j\dot{R}_j^2}{D^2} + \dfrac{R_j^2\left(R_j\dot{R}_i\dot{x}_j+5R_i\dot{R}_j\dot{x}_j \right)}{D^3}.
```


## Dimensionless variables

Let us introduce the dimensionless variables

```math
\tau = \dfrac{t}{T_{r}} = t \dfrac{\omega_r}{2\pi}
```

```math
y_i = \dfrac{R_i}{R_{0i}}
```

```math
y_i'=\dfrac{dy_i}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{R}_i}{R_{0i}}\left(\dfrac{2\pi}{\omega_r}\right)
```

```math
y_i''=\dfrac{dy_i'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{R}_i}{R_{0i}}\left(\dfrac{2\pi}{\omega_r}\right)^2
```

```math
\zeta_i = \dfrac{z_i}{\lambda}
```

```math
\zeta_i' = \dfrac{\zeta_i}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{z}_i}{\lambda_r}\left(\dfrac{2\pi}{\omega_r} \right)
```

```math
\zeta_i'' = \dfrac{\zeta_i'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{z}_i}{\lambda_r}\left(\dfrac{2\pi}{\omega_r} \right)^2
```

```math
\delta = D\lambda_r = |z_i - z_j|/\lambda_r = |\zeta_i-\zeta_j|
```

In the above quantities, $`\omega_r`$ is a reference angular frequency and $`\lambda_r`$ is the corresponding reference wavelength and $`R_{0i}`$ is the equilibrium radius of bubble $`i`$.

---

From the dimensionless quantities, the dimensional variables can be easily obtained:

```math
t=\tau\cdot\left(\dfrac{2\pi}{\omega_r}\right)
```

```math
R_i=y_i\cdot R_{0i}
```

```math
\dot{R}_i=y_i'\cdot R_{0i} \left(\dfrac{\omega_r}{2\pi}\right)
```

```math
\ddot{R}_i=y''\cdot R_{0i} \left(\dfrac{\omega_r}{2\pi}\right)^2
```

```math
z_i=\zeta_i \cdot \lambda_r
```

```math
\dot{z}_i=\zeta_i' \cdot \lambda_r\left(\dfrac{\omega_r}{2\pi}\right) =\xi \cdot c_L
```

```math
\ddot{z}_i=\zeta_i'' \cdot \lambda_r \left(\dfrac{\omega_r}{2\pi}\right)^2 = \xi \cdot c_L \left(\dfrac{\omega_r}{2\pi} \right)
```

```math
D = \delta \lambda_r
```

## Dimensionless Governing Equations

By substituting the dimensionless quantities into the Keller--Miksis equation and translational motion, the dimensionkless governing equations can be derived as

```math
D_{KMi} R_i \ddot{R}_i - G_{ij}^{(rad,impl)} = N_{KMi} +\dfrac{\dot{z}_i^2}{4}+ G_{ij}^{(rad,expl)}
```

```math
R_i\ddot{z}_i-3G_{ij}^{(trn,impl)} = \dfrac{3F_{ex}(z_i, t)}{2\pi \rho_L R_i^2} -3 \dot{R}\dot{z} + 3G_{ij}^{(trn,expl)}
```

---

```math
D_{KMi}R_{0i}^2\left(\dfrac{\omega_r}{2\pi}\right)^2 y_iy_i'' - G_{ij}^{(rad,impl)} = N_{KMi} + \dfrac{\zeta_i^2\lambda_r^2}{4}\left(\dfrac{\omega_r}{2\pi} \right)^2 + G_{ij}^{(rad,expl)}
```

```math
R_{0i}\lambda_r \left(\dfrac{\omega_r}{2\pi}\right)^2y_i\zeta_i''-G_{ij}^{(trn,impl)} = \dfrac{3F_{ex}(\zeta_i, t)}{2\pi \rho_L R_{0i}^2y_i^2} -3 \left(\dfrac{\omega_r}{2\pi}\right)^2R_{0i}\lambda_ry_i'\zeta_i'+G_{ij}^{(trn,expl)}
```

The first and secod equation is multiplied with $`\dfrac{1}{R_{0i}^2}\left( \dfrac{2\pi}{\omega_r} \right)^2`$ and $`\dfrac{1}{y_iR_{0i}\lambda_r}\left(\dfrac{2\pi}{\omega_r}\right)^2`$, respectively. The resulting dimensionless system is:

```math
\tilde{D}_{KMi}y_i'' - \tilde{G}_{ij}^{(rad,impl)}=\tilde{N}_{KMi} + \left(\dfrac{\lambda_r}{2R_{0i}}\right)^2\zeta_i^2 +\tilde{G}_{ij}^{(rad,expl)},
```

```math
\zeta_i''-\dfrac{1}{y_i} \tilde{G}_{ij}^{(trn,impl)}=\dfrac{2\pi3F_{ex}(\zeta_i\tau)}{\rho_L R_{0i}^3\lambda_r\omega_r^2y_i^3} -3\dfrac{y_i'\zeta_i'}{y_i} + \dfrac{1}{y_i}\tilde{G}_{ij}^{(trn,expl)},
```

where $`\tilde{D}_{KMi}`$ and $`\tilde{N}_{KMi}`$ are given in model descriptions (KM0D, KM1D) and the coupling terms for bubble-pair $`i\in(0, 1)`$ and $`j=(1-i)`$:

```math
\begin{align*}
\tilde{G}_{ij}^{(rad,impl)} &=\dfrac{1}{R_{0i}^2}\left(\dfrac{2\pi}{\omega_r}\right)^2G_{ij}^{(rad,impl)} \\
&= \dfrac{1}{R_{0i}^2}\left(\dfrac{2\pi}{\omega_r} \right)^2 \left(-\dfrac{R_j^2\ddot{R}_j}{D} + (-1)^i \dfrac{R_j^3\ddot{x}_j}{2D^2} \right) \\
&=\dfrac{R_{0j}^3}{R_{0i}^2 \lambda_r} \left(-\dfrac{y_j^2}{\delta}y_j'' + (-1)^i\dfrac{y_j^3}{2\delta^2}\zeta_j'' \right) \\
\\

\tilde{G}_{ij}^{(rad,expl)} &= \dfrac{1}{R_{0i}^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 G_{ij}^{(radn,expl)} \\
&=\dfrac{1}{R_{0i}^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 \left( -\dfrac{2R_j\dot{R}_j^2}{D} + (-1)^i\dfrac{R_j^2\left(\dot{z}_i\dot{R}_j+5\dot{R}_j \dot{z_j}\right)}{2D^2} - \dfrac{R_j^3\dot{z}_j\left(\dot{z}_i + 2 \dot{z}_j\right)}{2D^3}\right) \\
&=\dfrac{R_{0j}^3}{R_{0i}^2\lambda_r} \left(-\dfrac{2y_jy_j'^2}{\delta}+(-1)^i\dfrac{y_j^2\left(\zeta_i'y_j'+5y_j'\zeta_j' \right)}{2\delta^2} - \dfrac{y_j^3\zeta_j'\left(\zeta_i'+2\zeta_j' \right)}{2\delta^3} \right) \\
\\

\tilde{G}_{ij}^{(trn,impl)} &= \dfrac{3}{R_{0i}\lambda_r} \left(\dfrac{2\pi}{\omega_r}\right)^2 G_{ij}^{(trn,impl)} \\
& = \dfrac{3}{R_{0i}\lambda_r}\left(\dfrac{2\pi}{\omega_r}\right)^2 \left(-(-1)^i\dfrac{R_iR_j^2\ddot{R}_j}{D^2}+\dfrac{R_iR_j^3\ddot{x}_j}{D^3} \right) \\
&=\dfrac{3R_{0j}^3}{\lambda_r^3}\left(-(-1)^i\dfrac{y_iy_j^2}{\delta^2}y_j''+\dfrac{y_iy_j^3}{\delta^3}\zeta_j''\right) \\
\\

\tilde{G}_{ij}^{(trn,expl)} &= \dfrac{3}{R_{0i}\lambda_r} \left(\dfrac{2\pi}{\omega_r}\right)^2G_{ij}^{(trn,expl)} \\
& = \dfrac{3}{R_{0i}\lambda_r}\left(\dfrac{2\pi}{\omega_r}\right)^2 \left(-(-1)^i\dfrac{\dot{R}_iR_j^2\dot{R}_j+2R_iR_j\dot{R}_j^2}{D^2} + \dfrac{R_j^2\left(R_j\dot{R}_i\dot{x}_j+5R_i\dot{R}_j\dot{x}_j \right)}{D^3}\right) \\
& = \dfrac{3R_{0j}^3}{\lambda_r^3}\left(-(-1)^i\dfrac{y_i'y_j^2y_j'+2y_iy_jy_j'^2}{\delta^2} + \dfrac{y_j^2\left(y_jy_i'\zeta_j'+5y_iy_j'\zeta_j'\right)}{\delta^3} \right)
\end{align*}
```

## First-order system

The state vector defined as $`\mathbf{x}^T=[y_0, y_1, \zeta_0, \zeta_1, y_0', y_1', \zeta_0', \zeta_1']`$; thus, the first-order system is written as

```math
\left[\begin{array}{c} 
\\
I \\
\\
A(\tau) \\
\\
\end{array}\right]
\left[\begin{array}{c} 
y_0' \\
y_1' \\
\zeta_0' \\
\zeta_1' \\

y_0'' \\
y_1'' \\
\zeta_0'' \\
\zeta_1'' \\
\end{array}\right]=
\left[\begin{array}{c}
y_0'' \\
y_1'' \\
\zeta_0'' \\
\zeta_1'' \\
\dfrac{1}{D_{KM0}}\left(N_{KM0} + C_{11,0}\zeta_0'^2+\tilde{G}_{01}^{(rad,expl)}\right) \\
\dfrac{1}{D_{KM1}}\left(N_{KM1} + C_{11,1}\zeta_1'^2+\tilde{G}_{10}^{(rad,expl)}\right) \\
C_{12,0}\cdot F_{ex}(\zeta_0,\tau)\dfrac{3}{y_0^3} - \dfrac{1}{y_0}\left(3y_0'\zeta_0'+ \tilde{G}_{01}^{(trn,expl)} \right) \\
C_{12,1}\cdot F_{ex}(\zeta_1,\tau)\dfrac{3}{y_1^3} - \dfrac{1}{y_1}\left(3y_1'\zeta_1'+ \tilde{G}_{10}^{(trn,expl)} \right)
\end{array}\right]
```

where $`I`$ is an identy matrix and $`A(\tau)`$ is the time-dependent coupling matrix given as 

```math
A(\tau)=\left[\begin{array}{cccc}
1 & \dfrac{C_{13,0}}{D_{KM0}}\dfrac{y_1^2}{\delta} & 0 & -\dfrac{C_{13,0}}{D_{KM0}}\dfrac{y_1^3}{2\delta^2} \\
\dfrac{C_{13,1}}{D_{KM1}}\dfrac{y_0^2}{\delta} & 1 & \dfrac{C_{13,1}}{D_{KM1}}\dfrac{y_0^3}{2\delta^2} & 0 \\
0 & C_{14,0}\dfrac{y_0y_1^2}{\delta^2} & 1 & -C_{14,0}\dfrac{y_0y_1^3}{\delta^3} \\
-C_{14,1}\dfrac{y_1y_0^2}{\delta^2} & 0 & -C_{14,1}\dfrac{y_1y_0^3}{\delta^3} & 1 \\
\end{array}\right]
```