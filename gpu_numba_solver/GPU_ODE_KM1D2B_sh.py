import numpy as np
import time
from system_definitions import KM1D2B

from system_definitions.KM1D2B import CP, SP, DP, DEFAULT_MAT_PROPS, DEFAULT_EQ_PROPS, DEFAULT_SOLVER_OPTS
import GPU_ODE
GPU_ODE.setup(KM1D2B)
from GPU_ODE import SolverObject
import matplotlib.pyplot as plt


# GLOBAL (CONSTANT) PARAMETERS
MAT_PROPS   = DEFAULT_MAT_PROPS.copy()
EQ_PROPS    = DEFAULT_EQ_PROPS.copy()
SOLVER_OPTS = DEFAULT_SOLVER_OPTS.copy()

EQ_PROPS["R0"][0]   = 50.0 * 1e-6
EQ_PROPS["R0"][1]   = 50.0 * 1e-6
EQ_PROPS["FREQ"][0] = 25.0 * 1e3
EQ_PROPS["FREQ"][1] = 50.0 * 1e3
EQ_PROPS["REL_FREQ"]= 50.0 * 1e3


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
                       td: np.ndarray):
    
    problem_number = 0
    for p0 in pa0:
        EQ_PROPS["PA"][0] = p0 * 1e5
        for p1 in pa1:
            EQ_PROPS["PA"][1] = p1 * 1e5

            # Time Domain
            solver.set_host(problem_number, "time_domain", 0, td[0])
            solver.set_host(problem_number, "time_domain", 1, td[1])

            # Actual State
            solver.set_host(problem_number, "actual_state", 0, 1.0)     # Dimensionless Radius
            solver.set_host(problem_number, "actual_state", 1, 1.0)
            solver.set_host(problem_number, "actual_state", 2,  x0)     # Dimensionless Bubble Position
            solver.set_host(problem_number, "actual_state", 3, -x0)
            solver.set_host(problem_number, "actual_state", 4, 0.0)     # Dimensionless Bubble Wall Velocity
            solver.set_host(problem_number, "actual_state", 5, 0.0)     
            solver.set_host(problem_number, "actual_state", 6, 0.0)     # Dimensionless Bubble Translational Velocity
            solver.set_host(problem_number, "actual_state", 7, 0.0)     

            # Equation properties
            for (k, f) in CP.items():
                # i bubble index (0, 1)
                for i in range(2):
                    solver.set_host(problem_number, "control_parameters", i * 12 + k, f(i, **MAT_PROPS, **EQ_PROPS))

            # Acoustic field properties
            for (k, f) in DP.items():
                for i in range(EQ_PROPS["k"]):
                    solver.set_host(problem_number, "dynamic_parameters", i + k*EQ_PROPS["k"], f(i, **MAT_PROPS, **EQ_PROPS))

            problem_number +=1
        
    for (k, f) in SP.items():
        solver.set_shared_host("shared_parameters", k, f(**MAT_PROPS, **EQ_PROPS))    



if __name__ == "__main__":
    
    # Create the SolverObject
    solver = SolverObject(number_of_threads=SOLVER_OPTS["NT"],
                          system_dimension=SOLVER_OPTS["SD"],
                          number_of_control_parameters=SOLVER_OPTS["NCP"],
                          number_of_dynamic_parameters=SOLVER_OPTS["NDP"] * EQ_PROPS["k"],
                          number_of_shared_parameters=SOLVER_OPTS["NSP"],
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

    fill_solver_object(solver=solver,
                           pa0=pa0,
                           pa1=pa1,
                           x0=X0[0],
                           td=time_domain)
    

    print(solver._h_control_parameters)
    
    solver.syncronize_h2d("all")


