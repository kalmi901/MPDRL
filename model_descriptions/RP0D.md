# Model Description (RP0D)

- Rayleigh--Plesset bubble model (RP)
- Fix position, translational motion is not included (0D)
- Simple harmonic excitation

Note: This model is used only for debugging the GPU solver.


### References

[1] C.E., Brennen, Cavitation and bubble dynamics., Cambridge University Press (2014)

## Rayleigh--Plesset Equation
The dynamics of a spherical bubble in incompressible fluid is described by the Rayleigh--Plesset equation [1]:

```math 
R\ddot{R}+\dfrac{3}{2}\dot{R}^2=\dfrac{p_L-p(t)}{\rho}, 
```

where $`R`$ is the time dependent bubble radius, $`p_L`$ is the pressure at the bubble wall, and $`p(t)`$ is the far field pressure. According to ultrasonic irradiation, the far field pressure is

```math
p(t)=p_0 + p_{A}(t) = p_0 + P_{A}\cdot \sin(\omega t), 
```

where $`P_{A}`$ is the pressure amplitude and $`\omega`$ is the angular frequency of the excitation.
The liquid pressure at the bubble wall is 

```math
p_L = p_G(t) + p_v - \dfrac{2\sigma}{R} - 4\mu \dfrac{\dot{R}}{R},
```

where $`p_v`$ is the vapour pressure, $`\sigma`$ is the surface tension and $`\mu`$ is the liquid dynamic visosity and $`p_G`$ is the gas pressure. 
The gas pressure obeys a polytrophic state of change

```math
p_G(t)=p_{G0}\left(\dfrac{R_0}{R}\right)^{3\gamma}=\left(p_0 - p_v + \dfrac{2\sigma}{R_0} \right)\cdot\left(\dfrac{R_0}{R}\right)^{3\gamma},
```

where $`\gamma`$ is the polytrophic exponent.


### Dimensionless Variables

The governing equation is rewritten into dimensionless form by introducing the following dimensionless quantities:

```math
\tau = \dfrac{t}{T} = t \cdot \dfrac{\omega}{2\pi}
```

```math
y = \dfrac{R}{R_0}
```

```math
y'=\dfrac{dy}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\dot{R}}{R_0}\left(\dfrac{2\pi}{\omega}\right)
```

```math
y''=\dfrac{dy'}{dt} \cdot \dfrac{dt}{d\tau} = \dfrac{\ddot{R}}{R_0}\left(\dfrac{2\pi}{\omega}\right)^2
```

---

From the dimensionless quantities, the dimensional variables be easily obtained:

```math
R=y\cdot R_0
```

```math
\dot{R}=y'\cdot R_0 \left(\dfrac{\omega}{2\pi}\right)
```

```math
\ddot{R}=y''\cdot R_0 \left(\dfrac{\omega}{2\pi}\right)^2
```

---

## Dimensionless Rayleigh--Plesset equation

By substituting the dimensionless variables into the Rayleigh--Plesset equation, the dimensionless governing equation is derived.

```math
R_0^2 \left( \dfrac{\omega}{2\pi} \right)^2 yy'' + \dfrac{3}{2}R_0^2\left( \dfrac{\omega}{2\pi} \right)^2y'^2 = \dfrac{p_L - p(\tau)}{\rho}
```

```math
yy'' + \dfrac{3}{2}y'^2 = \dfrac{p_L - p(\tau)}{\rho R_0^2} \cdot \left( \dfrac{2\pi}{\omega} \right)^2
```

```math
p(\tau)=p_{0}+p_{A}(t(\tau)) = p_{0} + P_{A} \cdot \sin\left(\omega \cdot \dfrac{2\pi}{\omega} \tau\right) = p_0 + P_{A}\sin(2\pi \tau).
```

After expanding the right-hand side, one can introduce constant variables that can be pre-computed:

```math
\begin{align*}
\dfrac{p_L-p(\tau)}{\rho R_0^2} \cdot \left( \dfrac{2\pi}{\omega} \right)^2 &= \dfrac{1}{\rho R_0^2} \cdot \left( \dfrac{2\pi}{\omega} \right)^2 \left[p_{G0}\left( \dfrac{1}{y} \right)^{3\gamma} +(p_v-p_0) -\dfrac{2\sigma}{R_0}\cdot\dfrac{1}{y}-4\mu\cdot\dfrac{\omega}{2\pi}\cdot \dfrac{y'}{y} -p_A(\tau)\right] \\
&= C_0 + C_1\left(\dfrac{1}{y}\right)^{C_2} - C_3 \cdot \dfrac{1}{y} - C_4 \cdot \dfrac{y'}{y} - C_5\cdot\sin(2\pi\tau)
\end{align*}
```

where

```math
\begin{align*}
C_0 &= \dfrac{p_v-p_0}{\rho R_0^2}\cdot \left( \dfrac{2\pi}{\omega} \right)^2 \\

C_1 &= \dfrac{p_{G0}}{\rho R_0^2}\cdot \left( \dfrac{2\pi}{\omega} \right)^2 = \left(p_0 - p_v + \dfrac{2\sigma}{R_0} \right)\cdot\dfrac{1}{\rho R_0^2}\cdot \left( \dfrac{2\pi}{\omega} \right)^2 \\

C_2 &= 3\gamma \\

C_3 &= \dfrac{2\sigma}{\rho R_0^3}\cdot \left( \dfrac{2\pi}{\omega} \right)^2 \\

C_4 &= \dfrac{4\mu}{\rho R_0^2}\cdot \left( \dfrac{2\pi}{\omega} \right) \\

C_5 &= \dfrac{P_A}{\rho R_0^2}\cdot \left( \dfrac{2\pi}{\omega} \right)^2 \\
\end{align*}
```
## First-order ode system
For numerical computation, the dimensionless Rayleigh--Plesset equation is rewritten into a system of first-order differential equations. The state vector is defined as $`\mathbf{x}^T=[y, y']`$; thus, the first-order system is written as

```math
\begin{align*}
x_0'&=x_1 \\

x_1'&=\left( \dfrac{1}{x_0} \right)^{C_2}-C_3 \dfrac{1}{x_0} - C_4 \dfrac{x_1}{x_0}-C_5\sin(2\pi \tau) -\dfrac{3}{2}\dfrac{x_1^2}{x_0}
\end{align*}
```