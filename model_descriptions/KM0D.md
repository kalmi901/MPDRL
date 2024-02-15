# Model Description (KM0D)

- Keller--Miksis bubble model (KM)
- Fix position, translational motion is not included (0D)
- Harmonic excitation up to $k$ components

Note: The present model is the building block of the more advanced models applied in the present studies.

### References

[1] J.B. Keller, M. Miksis, Bubble oscillations of large amplitude, J. Acoust. Soc. Am., 68(2), (1980), pp. 628-633. \
[2] F. Hegedűs, K. Klapcsik, W. Lauterborn, U. Parlitz, R. Mettin, GPU accelerated study of a dual-frequency driven single bubble in a 6-dimensional parameter space: The active cavitation threshold, Ultrason. Sonochem., 67, (2020), p. 105067.


## Keller--Miksis equation

The volumetric oscillation of a spherical bubble in compressibe liquid is described by the Keller--Miksis equation [1]:

$\left(1 - \dfrac{\dot{R}}{c_L} \right)R\ddot{R} + \left(1 -\dfrac{\dot{R}}{3c_L}\right) \dfrac{3}{2}\dot{R}^2 = \left(1 + \dfrac{\dot{R}}{c_L} + \dfrac{R}{c_L}\dfrac{d}{dt}\right) \dfrac{p_L-p_{\infty}(t)}{\rho_L},$

where $R$ is the time dependent bubble radius, $\rho_L$ is the liquid density $c_L$ is the sound speed in the liquid domain and $p_L$ is the liquid pressure at the bubble wall. The far field pressure $p_{\infty}(t)$ depends on the acoustic irradation. In the present model the acousic field is composed of $k$ harmonic components discussed in the following subsection.

### Acoustic field

The acoustic field is the sum of the ambinent pressure $p_0$ and the pressure excitatation $p_A(t)$:

$p_\infty(t) = p_0 + p_A(t)$


The excitation pressure is a sum of $k$ harmonic components

$p_{A}(t) = \sum\limits_{i=0}^k P_{Ai} \cdot \sin(\omega_i t + \theta_i),$ 

where $P_{Ai}$ and $\omega_i$ is the pressure amplitude and the driving frequecy of component $i$, $\theta_i$ is the phase shift between the excitation components and $k$ is the number of harmonic components.

### Liquid pressure at the bubble wall

The liquid pressure at the bubble wall is written as

$p_L = p_G + p_v - \dfrac{2\sigma}{R} - 4\mu_L \dfrac{\dot{R}}{R},$

where $p_G$ is the partial pressure of the gas content and $p_v$ is the partial pressure of the vapour. $\sigma$ is the surface tension and $\mu_L$ is the liquid viscosity.

### Gas Pressure

The gas obeys a polythrophic relationship

$p_G = P_{G0}\left( \dfrac{R_0}{R}\right)^{3\gamma},$

where $\gamma=1.4$ is the politrophic exponent, $R_0$ is the reference pressure and $p_{G0}$ is the refrence gas pressure written as

$p_{G0}=\left(p_0 - p_v + \dfrac{2\sigma}{R_0} \right)$


## Rearrangement of the Keller--Miksis equation

First, the time derivatives are calculated.

$\dfrac{dp_{G}}{dt} = \dot{p}_{G}= -3\gamma p_{G0} \left( \dfrac{R_0}{R} \right)^{3\gamma-1}\dfrac{R_0}{R^2}\dot{R}=-3\gamma p_{G0} \left( \dfrac{R_0}{R} \right)^{3\gamma}\dfrac{\dot{R}}{R}$

$\dfrac{dp_{L}}{dt} = \dot{p}_L = \dot{p}_{G} + \dfrac{2\sigma}{R^2}\dot{R} + 4\mu_l\left( \dfrac{\dot{R}}{R}\right)^2 - 4\mu_L \dfrac{\ddot{R}}{R}$

$\dfrac{p_\infty}{dt}=\dot{p}_{\infty} = 0+ \dot{p}_{A}(t),$

where 

$\dot{p}_A(t) = \sum\limits_{i=0}^k\omega_i P_{Ai}\cdot\cos(\omega_i t + \theta_i),$ 

By substituting the above expressions into the original Keller--Miksis equation one obtains

$\left(1 - \dfrac{\dot{R}}{c_L} \right)R\ddot{R} + \left(1 -\dfrac{\dot{R}}{3c_L}\right)\dfrac{3}{2}\dot{R}^2 = \\
\left(1 + \dfrac{\dot{R}}{c_L}\right) \cdot \dfrac{1}{\rho_L} \left(p_{G0}\left(\dfrac{R_0}{R}\right)^{3\gamma}+p_v-\dfrac{2\sigma}{R} - 4\mu_L\dfrac{\dot{R}}{R} - p_0 - p_A(t) \right) \\
+\dfrac{R}{c_L\rho_L}\left(-3\gamma p_{G0}\left(\dfrac{R_0}{R} \right)^{3\gamma} \dfrac{\dot{R}}{R} + \dfrac{2\sigma}{R^2}\dot{R} + 4\mu_L \left(\dfrac{\dot{R}}{R}\right)^2- 4\mu_L\dfrac{\ddot{R}}{R} - \dot{p}_A(t)\right) $


Observe that some terms containing surface tension and viscosity could be simplified, then rearranging the equation yields


$\left(1 - \dfrac{\dot{R}}{c_L} + \dfrac{4\mu_L}{c_L\rho_LR}\right)R\ddot{R}= \left(1 + (1-3\gamma)\dfrac{\dot{R}}{c_L}\right)\cdot \dfrac{p_{G0}}{\rho_L}\left(\dfrac{R_0}{R}\right)^{3\gamma} - \left(1 + \dfrac{\dot{R}}{c_L}\right)\dfrac{p_0-p_v}{\rho_L} \\
-\dfrac{1}{\rho_L}\left( \dfrac{2\sigma}{R} + 4\mu_L \dfrac{\dot{R}}{R}\right) - \left(1 - \dfrac{\dot{R}}{3c_L}\right)\dfrac{3}{2}\dot{R}^2 -\left(1 + \dfrac{\dot{R}}{c_L}\right)\dfrac{p_{A}(t)}{\rho_L} - \dfrac{R \dot{p}_A(t)}{c_L\rho_L}$

---

The following simplified notations are introduced

$D_{KM} R\ddot{R} = N_{KM},$

where


$N_{KM}= 
\left(1 + \dfrac{\dot{R}}{c_L}\right) \cdot \dfrac{1}{\rho_L} \left(p_{G0}\left(\dfrac{R_0}{R}\right)^{3\gamma}+p_v-\dfrac{2\sigma}{R} - 4\mu_L\dfrac{\dot{R}}{R} - p_0 - p_A(t) \right) \\
+\dfrac{R}{c_L\rho_L}\left(-3\gamma p_{G0}\left(\dfrac{R_0}{R} \right)^{3\gamma} \dfrac{\dot{R}}{R} + \dfrac{2\sigma}{R^2}\dot{R} + 4\mu_L \left(\dfrac{\dot{R}}{R}\right)^2- 4\mu_L\dfrac{\ddot{R}}{R} - \dot{p}_A(t)\right) $

and

$D_{KM} = 1 - \dfrac{\dot{R}}{c_L} + \dfrac{4\mu_L}{c_L\rho_LR} $

### Dimensionless Variables

Let us introduce dimensionless quantities

$\tau = \dfrac{t}{T_{r}} = t \dfrac{\omega_r}{2\pi}$

$y = \dfrac{R}{R_0}$

$y'=\dfrac{dy}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{R}}{R_0}\left(\dfrac{2\pi}{\omega_r}\right)$

$y''=\dfrac{dy'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{R}}{R_0}\left(\dfrac{2\pi}{\omega_r}\right)^2$

In the above quantities $\omega_r$ is a reference angular frequency, tipically $\omega_r=\omega_0$, but we keep it as a free parameter during the derivation of the dimensionless systems.

---

From the dimensionless quantities the dimensional variables can be easily obtain:

$t=\tau\cdot\left(\dfrac{2\pi}{\omega_r}\right)$

$R=y\cdot R_0$

$\dot{R}=y'\cdot R_0 \left(\dfrac{\omega_r}{2\pi}\right)$

$\ddot{R}=y''\cdot R_0 \left(\dfrac{\omega_r}{2\pi}\right)^2$



## Dimensionless Keller--Miksis equation

Let us substitute the dimensionless variables into the modified form of the Keller--Miksis equation

$D_{KM} R_{0}^2\left( \dfrac{\omega_r}{2\pi} \right)^2 yy'' = N_{KM},$

then multiplying the equation with $\dfrac{1}{R_0^2}\left( \dfrac{2\pi}{\omega_r} \right)^2$ yields:

$\tilde{D}_{KM} y'' = \tilde{N}_{KM},$

where

$\tilde{N}_{KM}= \dfrac{1}{R_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 N_{KM} = \left(\dfrac{p_{G0}}{\rho_LR_{0}^2}\left(\dfrac{2\pi}{\omega_r} \right)^2 + (1-3\gamma)\dfrac{p_{G0}}{\rho_LR_0c_L}\dfrac{2\pi}{\omega_r}y' \right)\left(\dfrac{1}{y} \right)^{3\gamma} \\
-\dfrac{p_0-p_v}{\rho_LR_q^2}\left(\dfrac{2\pi}{\omega_r}\right)^2\left(1+ \dfrac{R_0}{c_L}\dfrac{\omega_R}{2\pi}y' \right) -\dfrac{2\sigma}{\rho_LR_0^3}\left( \dfrac{2\pi}{\omega_r}\right)^2 \dfrac{1}{y} - \dfrac{4\mu_L}{\rho_LR_0^2}\dfrac{2\pi}{\omega_r}\dfrac{y'}{y} \\
-\left(1 - \dfrac{R_0}{3c_L}\dfrac{\omega_r}{2\pi}y'\right) \dfrac{3}{2}y'^2 -\left(1 + \dfrac{R_0}{c_L}\frac{\omega_r}{2\pi}\right)\dfrac{1}{\rho_L R_0^2}\left(\dfrac{2\pi}{\omega_r} \right)^2p_{A}(\tau)-\dfrac{1}{\rho_Lc_LR_0}\left(\dfrac{2\pi}{\omega_r} \right)^2\dot{p}_{A}(\tau)y $

$\tilde{D}_{KM}=y D_{KM} =  y - \dfrac{R_0}{c_L}\dfrac{\omega_r}{2\pi}yy' + \dfrac{4\mu_L}{c_L\rho_LR_0} $

The pressure excitation and its time derivative are

$p_{A}(\tau) = \sum\limits_{i=0}^k P_{Ai} \cdot \sin\left(2\pi\dfrac{\omega_i}{\omega_r} \tau + \theta_i \right)$ 

$\dot{p}_{A}(\tau) = \sum\limits_{i=0}^k \omega_i P_{Ai} \cdot \cos\left(2\pi\dfrac{\omega_i}{\omega_r} \tau + \theta_i \right),$ 

respectively.

## First-order system

The above derived dimensionless Keller--Miksis equation is rewritten into a first order system and a set of pre-computed constans $C_k$ are introduced as in paper [2]. The state vector is $\mathbf{x}^T=[y, y']$; thus, the first-order system is

$x_0'=x_1$

$x_1'=\dfrac{\tilde{N}_{KM}}{\tilde{D}_{KM}}$

where 

$\tilde{N}_{KM}=\left(C_{0}+C_{1} x_1 \right)\left(\dfrac{1}{x_0}\right)^{C_8} - C_2 \left(1 +C_7 x_1\right) -C_3 \dfrac{1}{y_0}-C_4\dfrac{x_1}{y_0} \\
-\left(1 - \dfrac{C_7}{3}\right)\dfrac{3}{2}x_1^2 - (1 + C_7x_1)C_5p_A(\tau)-C_6\dot{p}_A(\tau)x_0,$

and

$\tilde{D}_{KM}=x_0-C_7x_0x_1+C_4C_7.$

The pressure excitation and its time derivative are

$p_{A}(\tau) =\sum\limits_{i=0}^k \cdot \sin\left(2\pi C_{9}C_{10+k+i} \tau + C_{10+2k+i} \right)$ 

$\dot{p}_{A}(\tau) = \sum\limits_{i=0}^k C_{10+i} C_{10+k+i} \cdot \cos\left(2\pi C_{9}C_{10+k+i} \tau + C_{10+2k+i} \right),$ 

respectively. Note that $k$ is the number of harmonic components.

### Pre-computed constants

#### Keller--Miksis

$C_0=\dfrac{p_{G0}}{\rho_L R_0^2}\left(\dfrac{2\pi}{\omega_r} \right)^2 = \left(p_0 -p_v +\dfrac{2\sigma}{R_0}\right) \dfrac{1}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2$

$C_1=\dfrac{(1-3\gamma)p_{G0}}{\rho_Lc_LR_0}=\dfrac{1-3\gamma}{\rho_Lc_LR_0}\left(p_0 -p_v +\dfrac{2\sigma}{R_0}\right)\dfrac{2\pi}{\omega_r}$

$C_2=\dfrac{p_0-p_v}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2$

$C_3=\dfrac{2\sigma}{\rho_LR_0^3}\left(\dfrac{2\pi}{\omega_r}\right)^2$

$C_4=\dfrac{4\mu_L}{\rho_LR_0^2}\dfrac{2\pi}{\omega_r}$

$C_5=\dfrac{1}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2$

$C_6=\dfrac{1}{\rho_Lc_LR_0}\left(\dfrac{2\pi}{\omega_r}\right)^2$

$C_7=\dfrac{R_0}{c_L}\dfrac{\omega_r}{2\pi}$

$C_8=3\gamma$

$C_{9}=1/\omega_r$

#### Excitation parameters

$C_{10+i}=P_{Ai}$

$C_{10+k+i}=\omega_i $

$C_{10+2k+i}=\theta_i$