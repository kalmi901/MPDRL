import numpy as np
import time
from GPU_ODE import SolverObject
import matplotlib.pyplot as plt


# GLOBAL (CONSTANT) PARAMETERS
RE   = 110                  # Equilbirum Radius (micron)
FREQ = [25.0, 50.0]         # Excittion frequencies (kHz)
REL_FREQ = 25.0             # Relative Frequency (kHz)
PA0  = [0.8]                # Pressure Amplitude 0 (min, max), (bar)
PA1  = [0.0]                # Pressure Amplitude 1 (min, max), (bar)
SCALE = "lin"
RES  = 1                   # Resolution 1 --> Thread single time-series curve
TIME_DOMAIN = [0.0, 1.0]   # Number of Acoustic cycles

# Initial Conditions 
X0 = [0.24]                 # Dimensionless Position x/λ_0 (-)
X_RES = 21                  # Resolution of the initial conditions

# Solver options
NT  = RES * RES # Number Of Threads
SD  = 6         # System Dimension (r, u, x, v, Fb1, Fd)
NCP = 27        # Number Of Control Parameters
NACC = 5        # Number Of Accessories
SOLVER = "RKCK45"
BLOCKSIZE = 64
ATOL = 1e-9
RTOL = 1e-9
ITERATIONS = 8000

# Material Properties
PV  = 0.0       # Vapour Pressure [Pa]
RHO = 998.0     # Liquid Density [kg/m**3]
ST  = 0.0725    # Surface Tension [N/m]
VIS = 0.001     # Liquod Viscosity [Pa s]
CL  = 1500.0    # Liquid Sound Speed
P0  = 1         # Ambient Pressure [bar]
PE  = 1.4       # Poytrophix Exponent


def fill_solver_object(solver: SolverObject,
                        pa0: np.ndarray,
                        pa1: np.ndarray,
                        x0: np.float64,
                        time_domain: np.ndarray):
    
    problem_number = 0
    for p0 in pa0:
        for p1 in pa1:

            # Time Domain
            solver.set_host(problem_number, "time_domain", 0, time_domain[0])
            solver.set_host(problem_number, "time_domain", 1, time_domain[1])

            # Actual State
            solver.set_host(problem_number, "actual_state", 0, 1.0)     # Dimensionless Radius
            solver.set_host(problem_number, "actual_state", 1, x0)      # Dimensionless Position
            solver.set_host(problem_number, "actual_state", 2, 0.0)     # Dimensionless Wall velocity
            solver.set_host(problem_number, "actual_state", 3, 0.0)     # Dimensionless Translational velocity
            solver.set_host(problem_number, "actual_state", 4, 0.0)     # Primary Bjerknes Force
            solver.set_host(problem_number, "actual_state", 5, 0.0)     # Drag Force

            # Equation properties 

            # Wave length
            l1 = CL / (FREQ[0] * 1000)
            l2 = CL / (FREQ[1] * 1000)
            lr = CL / (REL_FREQ * 1000)

            # Angular Frequency
            w1 = 2.0 * np.pi * (FREQ[0] * 1000)
            w2 = 2.0 * np.pi * (FREQ[1] * 1000)
            wr = 2.0 * np.pi * (REL_FREQ * 1000)

            # Convert to SI Units
            R0 = RE * 1e-6              # Equilbrium radius (m)
            Pinf = P0 * 1e5             # Ambinet Pressures (Pa)

            solver.set_host(problem_number, "control_parameter", 0, (2.0 * ST / R0 + Pinf - PV) * (2.0* np.pi / R0 / wr)**2.0 / RHO)
            solver.set_host(problem_number, "control_parameter", 1, (1.0 - 3.0*PE) * (2 * ST / R0 + Pinf - PV) * (2.0*np.pi / R0 / wr) / CL / RHO)
            solver.set_host(problem_number, "control_parameter", 2, (Pinf - PV) * (2.0 *np.pi / R0 / wr)**2.0 / RHO)
            solver.set_host(problem_number, "control_parameter", 3, (2.0 * ST / R0 / RHO) * (2.0 * np.pi / R0 / wr)**2.0)
            solver.set_host(problem_number, "control_parameter", 4, 4.0 * VIS / RHO / (R0**2.0) * (2.0* np.pi / wr))
            solver.set_host(problem_number, "control_parameter", 5, ((2.0 * np.pi / R0 / wr)**2.0) / RHO)
            solver.set_host(problem_number, "control_parameter", 6, ((2.0 * np.pi / wr)** 2.0) / CL / RHO / R0)
            solver.set_host(problem_number, "control_parameter", 7, R0 * wr / (2 * np.pi) / CL)
            solver.set_host(problem_number, "control_parameter", 8, 3.0 * PE)

            # Physical Parameters
            solver.set_host(problem_number, "control_parameter",  9, p0 * 1e5)
            solver.set_host(problem_number, "control_parameter", 10, p1 * 1e5)
            solver.set_host(problem_number, "control_parameter", 11, w1)
            solver.set_host(problem_number, "control_parameter", 12, w2)
            solver.set_host(problem_number, "control_parameter", 13, 0)                     # Phase shift
            solver.set_host(problem_number, "control_parameter", 14, R0)

            # Parameters for translation
            solver.set_host(problem_number, "control_parameter", 15, (l1 / R0)**2)
            solver.set_host(problem_number, "control_parameter", 16, (2.0 * np.pi) / RHO / R0 / l1 / (wr * R0)**2.0)
            solver.set_host(problem_number, "control_parameter", 17, 4 * np.pi / 3.0 * R0**3.0)
            solver.set_host(problem_number, "control_parameter", 18, 12 * np.pi * VIS * R0)

            # Acoustic field properties
            solver.set_host(problem_number, "control_parameter", 19, 2 * np.pi / l1)        # k1 wavenumber
            solver.set_host(problem_number, "control_parameter", 20, 2 * np.pi / l2)        # k2 wavenumber
            solver.set_host(problem_number, "control_parameter", 21, 1.0 / l1)              # wavelength
            solver.set_host(problem_number, "control_parameter", 22, 1.0 / l2)              # wavelength
            solver.set_host(problem_number, "control_parameter", 23, 1.0 / RHO / CL)        # Ac. Impedance 
            solver.set_host(problem_number, "control_parameter", 24, CL)                    # Reference velocity
            solver.set_host(problem_number, "control_parameter", 25, 1.0 / wr)              # Reference frequency
            solver.set_host(problem_number, "control_parameter", 26, lr)                    # Reference length

            problem_number += 1



if __name__ == "__main__":

    # Create the Solver Object
    solver = SolverObject(number_of_threads=NT,
                          system_dimension=SD, 
                          number_of_control_parameters=NCP,
                          number_of_accessories=NACC,
                          method=SOLVER,
                          threads_per_block=BLOCKSIZE,
                          abs_tol=ATOL,
                          rel_tol=RTOL,
                          system_definition="System_Definition_KM1D")    
    solver.time_step_min = 1e-10

    time_domain = np.array(TIME_DOMAIN, dtype=np.float64)
    pa0 = np.array(PA0)
    pa1 = np.array(PA1)

    # Data Containers
    t  = np.zeros((ITERATIONS, ), dtype=np.float64)
    x  = np.zeros((ITERATIONS, ), dtype=np.float64)
    v  = np.zeros((ITERATIONS, ), dtype=np.float64)
    fb = np.zeros((ITERATIONS, ), dtype=np.float64)
    fd = np.zeros((ITERATIONS, ), dtype=np.float64)

    fill_solver_object(solver=solver,
                           pa0=pa0,
                           pa1=pa1,
                           x0=X0[0],
                           time_domain=time_domain)
    
    solver.syncronize_h2d("all")

    run_starts = time.time()
    for ic in range(ITERATIONS):
        print(f"Iterations: {ic:.0f}")
        solver.solve_my_ivp()
        solver.syncronize_d2h("accessories")
        solver.syncronize_d2h("time_domain")

        t[ic]  = solver.get_host(0, "time_domain", 1)
        fb[ic] = solver.get_host(0, "accessories", 0)
        fd[ic] = solver.get_host(0, "accessories", 1)
        x[ic]  = solver.get_host(0, "accessories", 3)
        v[ic]  = solver.get_host(0, "accessories", 4)

    run_ends = time.time()
    print(f"The total runtime was {1000 * (run_ends-run_starts):.2f} ms")

       
    # Create my beautiful figures, please!

    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.plot(t, x, "k.-", markersize=2, linewidth=1)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$x/\lambda$")
    plt.grid("both")

    plt.subplot(2, 1, 2)
    plt.plot(t, v, "k.-", markersize=2, linewidth=1)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\bar{x}/c$")
    plt.grid("both")


    plt.figure(1)
    plt.plot(t, fb * 1e6, "b.-", markersize=2, linewidth=1, label=r"$F_{B1}$")
    plt.plot(t, fd * 1e6, "r.-", markersize=2, linewidth=1, label=r"$F_{D}$")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$F,\, \mathrm{\mu m}$")
    plt.legend()
    plt.grid("both")

    plt.show()
