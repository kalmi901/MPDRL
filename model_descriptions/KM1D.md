# Model Description (KM1D)

- Keller--Miksis bubble model (KM)
- 1 dimensional translational motion (1D)
- Harmonic excitation up to $k$ components (Standing waves)

Note: The present model is the extension of model KM0D with 1-dimensional translational motion in a standing acoustic wave field. 

### References

[1] J.B. Keller, M. Miksis, Bubble oscillations of large amplitude, J. Acoust. Soc. Am., 68(2), (1980), pp. 628-633. \
[2] F. Hegedűs, K. Klapcsik, W. Lauterborn, U. Parlitz, R. Mettin, GPU accelerated study of a dual-frequency driven single bubble in a 6-dimensional parameter space: The active cavitation threshold, Ultrason. Sonochem., 67, (2020), p. 105067. \
[3] R.S. Meyer, M.L. Billet, J.W. Holl, Freestream Nuclei and Traveling-Bubble Cavitation, J. Fluids Eng., 114(4), (1992), pp. 672-679. \
[4] A.A. Doinikov, Translational motion of a spherical bubble in an acoustic standing wave of high intensity, Phys. Fluids, 14, (2002), pp. 1420-1425. \
[5] J.A. Reddy, A.J. Szeri, Coupled dynamics of translation and collapse of acoustically driven microbubbles, J. Acoust. Soc. Am., 112(4), (2002), pp. 1346-1352.

## Keller--Miksis equation

The volumetric oscillation of the bubble is described by the Keller--Miksis equation [1]:

```math
\left(1 - \dfrac{\dot{R}}{c_L} \right)R\ddot{R} + \left(1 -\dfrac{\dot{R}}{3c_L}\right) \dfrac{3}{2}\dot{R}^2 = \left(1 + \dfrac{\dot{R}}{c_L} + \dfrac{R}{c_L}\dfrac{d}{dt}\right) \dfrac{p_L-p_{\infty}(z,t)}{\rho_L} + \dfrac{u^2}{4},
```

where $`R`$ is the bubble radius, $`z`$ is the position of the bubble, $`\rho_L`$ is the liquid density, $`c_L`$ is the sound speed in the liquid domain, $`p_L`$ is the liquid pressure at the bubble wall and $`p_{\infty}(z, t)`$ is the excitation pressure calculated at the centre of the bubble ($`z`$).
Note that the feedback term ($`u^2/4`$) from the translational motion is added to the right-hand side [4].
First, the equation is rearranged to omit the time derivatives $`d/dt`$ on the right-hand side (for details see the model description of KM0D). The simplified form is given as:

```math
D_{KM} R\ddot{R} = N_{KM} + \dfrac{u^2}{4},
```

where

```math
\begin{split}
N_{KM}= 
\left(1 + \dfrac{\dot{R}}{c_L}\right) \cdot \dfrac{1}{\rho_L} \left(p_{G0}\left(\dfrac{R_0}{R}\right)^{3\gamma}+p_v-\dfrac{2\sigma}{R} - 4\mu_L\dfrac{\dot{R}}{R} - p_0 - p_A(x,t) \right) \\
+\dfrac{R}{c_L\rho_L}\left(-3\gamma p_{G0}\left(\dfrac{R_0}{R} \right)^{3\gamma} \dfrac{\dot{R}}{R} + \dfrac{2\sigma}{R^2}\dot{R} + 4\mu_L \left(\dfrac{\dot{R}}{R}\right)^2- 4\mu_L\dfrac{\ddot{R}}{R} - \dot{p}_A(x,t)\right) 
\end{split}
```

and

```math
D_{KM} = 1 - \dfrac{\dot{R}}{c_L} + \dfrac{4\mu_L}{c_L\rho_LR}.
```

The equation of translational motion is coupled to the Keller--Miksis bubble model to describe the translational bubble motion.

## Equation of translational bubble motion

Newton's second law of motion governs the translation of the bubble

```math
m_b\ddot{z}=F_{B1}+F_{D}+F_{m}+F_{g}+F_{h},
```

where $`m_b=(4/3)\pi R^3\rho_b`$ is the mass of the bubble, and $`x`$ is the bubble position. The terms on the right-hand side represent the forces acting on the bubble, including the primary Bjerknes force $`F_{B1}`$, the drag force $`F_{D}`$, the added mass force $`F_{m}`$ and the gravitational force $`F_{g}`$.

The added mass force (or virtual mass force) results from the inertia of the surrounding liquid, and it is significant when the density of the surrounding liquid is greater than the density of the accelerating particle (i.e., gas particle in water). The virtual mass force is written as [3]:

```math
F_{m}=-\dfrac{\rho_L}{2}\dfrac{d}{dt}\left[V(u-v)\right]\approx-\dfrac{\rho_L}{2}V\ddot{x}-\dfrac{3}{2}\dfrac{\rho_L}{R}V\dot{R}\dot{x},
```

where $`V=(4/3)\pi R^3`$ is the bubble volume, $`u=\dot{x}`$ is the translational velocity of the bubble. $`v`$ is the velocity of the surrounding liquid that is negligible in the present case.
The first term is the conventional form of the added mass. The second, lesser-known term appears due to the oscillation of the bubble.

By substituting $`F_m`$ into the equation of motion and rearranging one obtains

```math
\left(m_b + \dfrac{1}{2}\rho_LV \right) \ddot{z}+\dfrac{3}{2}\dfrac{\rho_LV}{R}\dot{R}\dot{z}=F_{ex}(z,t),
```

where $`F_{ex}(x,t) = F_{B1}+F_{D}+F_{g}+F_{h}`$ is the sum of external force without the virtual mass force. The bubble mass on the left-hand side is negligible since $`\rho_b\ll\rho_L`$  The above equation can be rearranged in the form of 

```math
R\ddot{z}+3\dot{R}\dot{z}=\dfrac{3F_{ex}(z,t)}{2\pi\rho_LR^2},
```

which is obtained in papers [4] using the Lagrangian formalism for the derivation. 

### External forces

Although the present implementation does not include all external forces, they are reviewed here for later additions to the model.

#### <ins> Primary Bjerknes Force </ins>

The Primary Bjerknes force (acoustic radiation force) is induced by the acoustic pressure gradient

```math
F_{B1}=-V\nabla p_{A}(x,t)=-\dfrac{4}{3}\pi R^3 \nabla p_{A}(x,t),
```

where $`\nabla p_{A}(x,t)`$ is the pressure gradient calculated at the centre of the bubble.

#### <ins> Drag Force </ins>

For the present study, the drag force is calculated as

```math
F_D=12\pi\mu_LR\left(\dot{x}-v_{ac}(z,t) \right),
```

where $`v_{ac}(x,t)`$ is the liquid velocity induced by the acoustic irradiation. \
Alternatively, the drag force can be calculated 

```math
F_D=\dfrac{1}{2}C_D\rho_L\dot{z}|\dot{z}|R^2.
```

The drag coefficient $`C_D`$ is estimated by empirical formulation

```math
C_D=\dfrac{24}{Re_b}\left(1 + 0.197Re_b^{0.63} + 2.6\cdot10^{-4}Re_b^{1.38}\right),
```

where $`Re_b=2|\dot{z}-u_{ac}|R\rho_L/\mu_L`$ is the bubble Reynolds number based on the relative motion between the bubble and the liquid. 

#### <ins> Gravitational force </ins>

The total gravitational force acting on the bubble is

```math
F_g=(\rho_b -\rho_L) \dfrac{4}{3}\pi R^3 g \approx -\rho_L \dfrac{4}{3}\pi R^3 g,
```

which is neglected in the present study.


#### <ins> History or Basset force </ins>

This force arises from the effect of the wake behind the bubble

```math
F_{h} = 8\pi \mu \int_{0}^{t}w(s,t) \frac{d}{ds}\left[R(s)U(s)\right]ds,
```

```math
w(s,t) = \exp \left[9\nu \int_{s}^{t}\frac{1}{R(s')^2ds'}\right] \textrm{erfc} \left[\sqrt{\int_{s}^{t}\frac{1}{R(s')^2ds'}}\right].
```

This force is significant when both the translational Reynolds number $`Re_t=R|\dot{z}|/\nu\ll1`$ and the radial Reynolds number $`Re_r=R|\dot{R}|/\nu\ll1`$ are small [5]; thus, in the present study, where high-amplitude collapse-like oscillations (high $`Re_r`$ values) are expected. 


## Acoustic field

The acoustic field is the sum of the ambient pressure $`p_0`$ and the excitation pressure $`p_A(x,t)`$

```math
p_\infty(x,t) = p_0 + p_A(z, t),
```

where the excitation is a sum of $`k`$ <b>standing waves</b>.

More realistic acoustic fields can be implemented later, but for now, we work with simple analytical formulas.

Two different forms of wave field are implemented depending on the origin of the acoustic field.
In the first case, an antinode is located at $`x=0`$. In the second case, the origin is a node.

The parameters of the acoustic field are the pressure amplitude $`P_{Ai}`$, the angular frequency $`\omega_i=2\pi f_i`$, the wave number $`k_i=2\pi/\lambda_i`$ and the phase shift $`\theta_i`$. The index $`i`$ denotes the component.

<ins>Antinode at the origin</ins>

```math
\begin{align*}
&p_A(z, t) = \sum_{i=0}^kP_{Ai} \cdot\cos(k_i z + \theta_i)\cdot\sin(\omega_i t + \theta_i) \\

&\dot{p}_A(z,t) = \sum_{i=0}^k \omega_i P_{Ai} \cdot \cos(k_iz + \theta_i)\cdot \cos(\omega_it+\theta_i) \\

&\nabla p_A(z,t) = -\sum_{i=0}^k k_iP_{Ai}\cdot\sin(k_iz+\theta_i) \cdot\sin(\omega_i t + \theta_i) \\

&u_{ac}(z, t)=-\dfrac{1}{\rho_L c_L} \sum_{i=0}^k P_{Ai}\cdot \sin(k_iz + \theta_i) \cdot \cos(\omega_i t + \theta_i)
\end{align*}
```

<ins>Node at the origin</ins>

```math
\begin{align*}
&p_A(z, t) = \sum_{i=0}^kP_{Ai} \cdot\sin(k_i z + \theta_i)\cdot\sin(\omega_i t + \theta_i) \\

&\dot{p}_{Ai}(z,t)= \sum_{i=0}^k \omega_i P_{Ai} \cdot \sin(k_iz + \theta_i)\cdot \cos(\omega_it+\theta_i) \\

&\nabla p_A(z,t) = \sum_{i=0}^k k_iP_{Ai}\cdot\cos(k_iz+\theta_i) \cdot\sin(\omega_i t + \theta_i) \\

&u_{ac}(z, t)=\dfrac{1}{\rho_L c_L} \sum_{i=0}^k P_{Ai}\cdot \cos(k_iz + \theta_i) \cdot \cos(\omega_i t + \theta_i)
\end{align*}
```

#### <ins>Mathematical description of the standing wave </ins>

This derivation is somewhat trivial, so only a brief reminder is provided here.

A standing wave is the superposition of two identical counter-propagating waves

```math
p_{1}(z,t) = \dfrac{1}{2}P_{A} \sin(\omega t + kz + 2\theta)
```

and 

```math
p_{2}(z,t) = \dfrac{1}{2}P_{A} \sin(\omega t - kz),
```

where $`2\theta`$ is the phase shift between the components.

The superposition of the waves

```math
p(z,t)=p_1 + p_2 = \dfrac{1}{2}P_A \left(\sin(\omega t + kz +2\theta) + \sin(\omega t - kz)  \right).
```

One can use the trigonometric identity (Sum-to-product identity)

```math
\sin \phi_1 + \sin \phi_2 = 2 \sin\left(\dfrac{\phi_1+\phi_2}{2}\right)\cos\left(\dfrac{\phi_1-\phi_2}{2} \right);
```

thus, the resulting wave is given by

```math
p(x,t) = P_{A}\cdot \cos(kz + \theta) \cdot \sin(\omega t + \theta)
```

## Dimensionless variables

Let us introduce dimensionless quantities

```math
\tau = \dfrac{t}{T_{r}} = t \dfrac{\omega_r}{2\pi}
```

```math
y = \dfrac{R}{R_0}
```

```math
y'=\dfrac{dy}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{R}}{R_0}\left(\dfrac{2\pi}{\omega_r}\right)
```

```math
y''=\dfrac{dy'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{R}}{R_0}\left(\dfrac{2\pi}{\omega_r}\right)^2
```

```math
\zeta = \dfrac{z}{\lambda}
```

```math
\zeta' = \dfrac{\zeta}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{z}}{\lambda_r}\left(\dfrac{2\pi}{\omega_r} \right)
```

```math
\zeta'' = \dfrac{\zeta'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{z}}{\lambda_r}\left(\dfrac{2\pi}{\omega_r} \right)^2
```

In the above quantities, $`\omega_r`$ is a reference angular frequency and $`\lambda_r`$ is the corresponding reference wavelength. Typically $`\omega_r=\omega_0`$, but we keep it as a free parameter during the derivation of the dimensionless systems.

---

From the dimensionless quantities, the dimensional variables can be easily obtained:

```math
t=\tau\cdot\left(\dfrac{2\pi}{\omega_r}\right)
```

```math
R=y\cdot R_0
```

```math
\dot{R}=y'\cdot R_0 \left(\dfrac{\omega_r}{2\pi}\right)
```

```math
\ddot{R}=y''\cdot R_0 \left(\dfrac{\omega_r}{2\pi}\right)^2
```

```math
z=\xi \cdot \lambda_r
```

```math
\dot{z}=\xi' \cdot \lambda_r\left(\dfrac{\omega_r}{2\pi}\right) =\xi \cdot c_L
```

```math
\ddot{z}=\xi'' \cdot \lambda_r \left(\dfrac{\omega_r}{2\pi}\right)^2 = \xi \cdot c_L \left(\dfrac{\omega_r}{2\pi} \right)
```

## Dimensionless Governing Equations

By substituting the dimensionless variables into the modified form of the Keller--Miksis equation and the equation of translational bubble motion one can obtain:

```math
D_{KM} R_{0}^2\left( \dfrac{\omega_r}{2\pi} \right)^2 yy'' = N_{KM} + \dfrac{\zeta'^2 \lambda_r^2}{4}\left( \dfrac{\omega_r}{2\pi}\right)^2
```

```math
R_0\lambda_r\left(\dfrac{\omega_r}{2\pi}\right)^2 y\zeta''+3R_0\lambda_r \left(\dfrac{\omega_r}{2\pi}\right)^2y'\zeta' = \dfrac{3F_{ex}(\zeta,\tau)}{2\pi\rho_LR_0^2y^2}
```

The first and secod equation is multiplied with $`\dfrac{1}{R_0^2}\left( \dfrac{2\pi}{\omega_r} \right)^2`$ and $`\dfrac{1}{yR_0\lambda_r}\left( \dfrac{2\pi}{\omega_r}\right)^2`$, respectively. The resulting dimensionless system is:

```math
\tilde{D}_{KM} y'' = \tilde{N}_{KM} + \left(\dfrac{\lambda_r}{2R_0}\right)^2 \cdot \zeta'^2,
```

```math
\zeta''+\dfrac{3y'\zeta'}{y}=\dfrac{2\pi 3 F_{ex}(\zeta, \tau)}{\rho_L R_0^3 \lambda_r \omega_r^2 y^3},
```

where $`\tilde{N}_{KM}= \dfrac{1}{R_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 N_{KM}`$ and $`\tilde{D}_{KM}=y D_{KM}`$. Their expanded forms are written in the model description (KM0D); thus, it is not repeated here. 


## First-order system

The state vector is $`\mathbf{x}^T=[y, \zeta, y', \zeta']`$; thus, the first order system is written as

```math
\begin{align*}
x_0' &= x_1 \\
x_1' &= x_2 \\
x_2' &= \dfrac{1}{\tilde{D}_{KM}}\left(\tilde{N}_{KM} + C_{11}x_3^2\right) \\
x_3' &= C_{12} \cdot F_{ex}(x_1, \tau) \dfrac{3}{x_0^3} - 3\dfrac{x_3 x_2}{x_0}
\end{align*}
```

where 

```math
\begin{align*}
&\tilde{N}_{KM}=\left(C_{0}+C_{1} x_2 \right)\left(\dfrac{1}{x_0}\right)^{C_8} - C_2 \left(1 +C_7 x_2\right) -C_3 \dfrac{1}{x_0}-C_4\dfrac{x_2}{x_0} \\
&-\left(1 - \dfrac{C_7}{3}\right)\dfrac{3}{2}x_2^2 - (1 + C_7x_2)C_5p_A(x_1, \tau)-C_6\dot{p}_A(x_1, \tau)x_0, \\

&\tilde{D}_{KM}=x_0-C_7x_0x_2+C_4C_7, \\

&F_{ex}(\zeta, \tau) = F_{B1} + F_{D}, \\

&F_{B1} = -C_{13} \cdot x_0^3 \cdot \nabla p_A(\zeta, \tau), \\

&F_{D} =-C_{14}\cdot x_0\left(C_{15}x_3 -u_{ac}(x_1, \tau) \right).
\end{align*}
```

---

```math
\begin{align*}
&p_A(x_1, \tau) = \sum_{i=0}^kC_{17+i} \cdot\cos(2\pi C_{10}C_{17+3k+i} x_1 + C_{17+2k+i})\cdot\sin(\omega_i \tau + C_{17+2k+i})\\

&\dot{p}_A(x_1, \tau) = \sum_{i=0}^k \omega_i C_{17+i} \cdot \cos(2\pi C_{10}C_{17+3k+i}x_1 + C_{17+2k+i})\cdot \cos(\omega_it+C_{17+2k+i})\\

&\nabla p_A(x_1, \tau) = -\sum_{i=0}^k k_iC_{17+i}\cdot\sin(2\pi C_{10}C_{17+3k+i}x_1+C_{17+2k+i}) \cdot\sin(\omega_i \tau + C_{17+2k+i})\\

&u_{ac}(x_1, \tau)=-C_{16}\sum_{i=0}^k C_{17+i}\cdot \sin(2\pi C_{10}C_{17+3k+i}x_1 + C_{17+2k+i}) \cdot \cos(\omega_i \tau + C_{17+2k+i})\\
\end{align*}
```

### Pre-computed constants

```math
\begin{align*}
\textbf{Keller--Miksis:} \\
\\

C_0&=\dfrac{p_{G0}}{\rho_L R_0^2}\left(\dfrac{2\pi}{\omega_r} \right)^2 = \left(p_0 -p_v +\dfrac{2\sigma}{R_0}\right) \dfrac{1}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 \\

C_1&=\dfrac{(1-3\gamma)p_{G0}}{\rho_Lc_LR_0}=\dfrac{1-3\gamma}{\rho_Lc_LR_0}\left(p_0 -p_v +\dfrac{2\sigma}{R_0}\right)\dfrac{2\pi}{\omega_r} \\

C_2&=\dfrac{p_0-p_v}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 \\

C_3&=\dfrac{2\sigma}{\rho_LR_0^3}\left(\dfrac{2\pi}{\omega_r}\right)^2 \\

C_4&=\dfrac{4\mu_L}{\rho_LR_0^2}\dfrac{2\pi}{\omega_r} \\

C_5&=\dfrac{1}{\rho_LR_0^2}\left(\dfrac{2\pi}{\omega_r}\right)^2 \\

C_6&=\dfrac{1}{\rho_Lc_LR_0}\left(\dfrac{2\pi}{\omega_r}\right)^2 \\

C_7&=\dfrac{R_0}{c_L}\dfrac{\omega_r}{2\pi} \\

C_8&=3\gamma \\

C_{9}&=1/\omega_r \\

\\
\textbf{Equation of motion:} \\
\\

C_{10}&=\dfrac{\lambda_r}{2\pi}=1/k_r \\

C_{11}&=\left(\dfrac{\lambda_r}{2R_0}\right)^2 \\

C_{12}&=\dfrac{2\pi}{\rho_LR_0^3\lambda_r\omega_r^2} \\

C_{13}&=\dfrac{4}{3}\pi R_0^3 \\

C_{14}&=12 \pi \mu_L R_0 \\

C_{15}&=c_L \\

C_{16}&=\dfrac{1}{\rho_L c_L} \\

\\
\textbf{Acoustic Field:} \\
\\

C_{17+i}&=P_{A,i} \\

C_{17+k+i}&=\omega_i \\

C_{17+2k+i}&=\theta_i \\

C_{17+3k+i}&=\dfrac{2\pi f_i}{c_L}=\dfrac{2\pi}{\lambda_i}=k_i \\
\end{align*}
```