import numpy as np
import time
from system_definitions import KM1D_d
from system_definitions.KM1D_d import CP, DEFAULT_MAT_PROPS, DEFAULT_EQ_PROPS, DEFAULT_SOLVER_OPTS
import GPU_ODE
GPU_ODE.setup(KM1D_d)
from GPU_ODE import SolverObject
import matplotlib.pyplot as plt


# GLOBAL (CONSTANT) PARAMETERS
MAT_PROPS   = DEFAULT_MAT_PROPS.copy()
EQ_PROPS    = DEFAULT_EQ_PROPS.copy()
SOLVER_OPTS = DEFAULT_SOLVER_OPTS.copy()

EQ_PROPS["R0"]      = 50.0 * 1e-6
EQ_PROPS["FREQ"][0] = 25.0 * 1e3
EQ_PROPS["FREQ"][1] = 50.0 * 1e3
EQ_PROPS["REL_FREQ"] = 50.0 * 1e3


PA0  = [0.2]                # Pressure Amplitude 0 (min, max), (bar)
PA1  = [0.0]                # Pressure Amplitude 1 (min, max), (bar)
SCALE = "lin"
RES  = 1                   # Resolution 1 --> Thread single time-series curve
TIME_DOMAIN = [0.0, 1.0]   # Number of Acoustic cycles

# Initial Conditions 
X0 = [0.24]                 # Dimensionless Position x/λ_0 (-)

ITERATIONS = 8000


def fill_solver_object(solver: SolverObject,
                        pa0: np.ndarray,
                        pa1: np.ndarray,
                        x0: np.float64,
                        time_domain: np.ndarray):
    
    problem_number = 0
    for p0 in pa0:
        EQ_PROPS["PA0"] = p0 * 1e5
        for p1 in pa1:
            EQ_PROPS["PA1"] = p1 * 1e5

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
            for (k, f) in CP.items():
                solver.set_host(problem_number, "control_parameters", k, f(**MAT_PROPS, **EQ_PROPS))


            problem_number += 1



if __name__ == "__main__":

    # Create the Solver Object
    solver = SolverObject(number_of_threads=SOLVER_OPTS["NT"],
                          system_dimension=SOLVER_OPTS["SD"], 
                          number_of_control_parameters=SOLVER_OPTS["NCP"],
                          number_of_accessories=SOLVER_OPTS["NACC"],
                          method=SOLVER_OPTS["SOLVER"],
                          threads_per_block=SOLVER_OPTS["BLOCKSIZE"],
                          abs_tol=SOLVER_OPTS["ATOL"],
                          rel_tol=SOLVER_OPTS["RTOL"])    

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
