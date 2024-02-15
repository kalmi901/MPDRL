# Model Description (KM1D2B)

- Keller--Miksis bubble model (KM)
- 1 dimensional translational motion (1D)
- 2 Bubbles (2B)
- Harmonic excitation up to $k$ components (Standing waves)

Note: The present model is the extension of model KM1D. 

### Referemces

[1] asdasda \
[2] asdadada \
[3]

## Keller--Miksis equation

The volumetric oscillation of the bubbles is described by the Keller--Miksis equation

$\left(1 - \dfrac{\dot{R}_i}{c_L} \right)R_i\ddot{R}_i + \left(1 -\dfrac{\dot{R}_i}{3c_L}\right) \dfrac{3}{2}\dot{R}_i^2 = \left(1 + \dfrac{\dot{R}_i}{c_L} + \dfrac{R_i}{c_L}\dfrac{d}{dt}\right) \dfrac{p_{Li}-p_{\infty}(z_i,t)}{\rho_L} + \dfrac{u_i^2}{4} + G_{ij}^{(rad)}, $

where $R_i$, and $z_i$ is the time dependent radius, and position of the bubble $i$. The bubble index $i\in(0, 1)$ and $j=1-i$. The last term $G_{ij}^{(rad)}$ is the couling term describing the effect of bubble $j$ on the radial oscillation of bubble $i$. The coupling terms are written as

$G_{ij}^{(rad)} = -\dfrac{R_j^2\ddot{R}_j + 2R_j\dot{R}_j^2}{D} + (-1)^i\dfrac{R_j^2\left(\dot{z}_i\dot{R}_j+R_j\ddot{z}_j +5\dot{R}_j \dot{z_j}\right)}{2D^2} - \dfrac{R_j^3\dot{z}_j\left(\dot{z}_i + 2 \dot{z}_j\right)}{2D^3} + \mathcal{O}\left(\dfrac{1}{D^4} \right), $

where $D = |z_j - z_i|$ is the distance between the center of the bubbles.

The coupling term can be decomposed into implicit and explicit ones $G_{ij}^{(rad)} = G_{ij}^{(rad, impl)} + G_{ij}^{(rad, expl)}$. Implicit terms contain second-order derivatives, explicit terms do not.

$G_{ij}^{(rad, impl)} = -\dfrac{R_j^2\ddot{R}_j}{D} + (-1)^i \dfrac{R_j^3\ddot{x}_j}{2D}$

$G_{ij}^{(rad, expl)} = -\dfrac{2R_j\dot{R}_j^2}{D} + (-1)^i\dfrac{R_j^2\left(\dot{z}_i\dot{R}_j+5\dot{R}_j \dot{z_j}\right)}{2D^2} - \dfrac{R_j^3\dot{z}_j\left(\dot{z}_i + 2 \dot{z}_j\right)}{2D^3}$

## Translational motion

The governing equation describing the translational bubble motion is

$R_i\ddot{z}_i+3\dot{R}_i\dot{z}_i=\dfrac{3F_{ex}(z_i,t)}{2\pi\rho_LR_i^2} + G_{ij}^{trn},$

where $G_{ij}^{(trn)}$ is the coupling term that describes the effect of bubble $j$ on the translational motion of bubble $i$. This coupling term is wrtitten as

$G_{ij}^{(trn)}=-(-1)^i\dfrac{1}{D^2}\dfrac{d}{dt}\left(R_iR_j^2\dot{R}_j^2\right) + \dfrac{R_j^2\left(R_iR_j\ddot{x}_j+R_j\dot{R}_i\dot{x}_j+5R_i\dot{R}_j\dot{x}_j \right)}{D^3} + \mathcal{O}\left(\dfrac{1}{D^4} \right)$.

After calculating the time derivative, this coupling term can also be decompies into explicit and implicit terms:

$G_{ij}^{(trn,impl)}=-(-1)^i\dfrac{R_iR_j^2\ddot{R}_j}{D^2}+\dfrac{R_iR_j^3\ddot{x}_j}{D^3}$

$G_{ij}^{(trn,expl)}=-(-1)^i\dfrac{\dot{R}_iR_j^2\dot{R}_j+2R_iR_j\dot{R}_j^2}{D^2} + \dfrac{R_j^2\left(R_j\dot{R}_i\dot{x}_j+5R_i\dot{R}_j\dot{x}_j \right)}{D^3}$