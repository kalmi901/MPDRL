import numpy as np
import pandas as pd
import time
from system_definitions import KM1D2B

from system_definitions.KM1D2B import CP, SP, DP, DEFAULT_MAT_PROPS, DEFAULT_EQ_PROPS, DEFAULT_SOLVER_OPTS
import GPU_ODE
GPU_ODE.setup(KM1D2B, k=2)
from GPU_ODE import SolverObject
import matplotlib.pyplot as plt


# GLOBAL (CONSTANT) PARAMETERS
MAT_PROPS   = DEFAULT_MAT_PROPS.copy()
EQ_PROPS    = DEFAULT_EQ_PROPS.copy()
SOLVER_OPTS = DEFAULT_SOLVER_OPTS.copy()

EQ_PROPS["R0"][0]   =  6.0 * 1e-6
EQ_PROPS["R0"][1]   =  5.0 * 1e-6
EQ_PROPS["FREQ"][0] = 20.0 * 1e3
EQ_PROPS["FREQ"][1] = 25.0 * 1e3
EQ_PROPS["REL_FREQ"]= 20.0 * 1e3
EQ_PROPS["k"]       = 2


PA0  = [-1.2 * 1.013]       # Pressure Amplitude 0, (bar)
PA1  = [0.0, 0.0, 0.0]      # Pressure Amplitude 1, (bar)
SCALE = "lin"
SOLVER_OPTS['NT'] = len(PA0) * len(PA1)      # Resolution 1 --> Thread single time-series curve
TIME_DOMAIN = [0.0, 1.0]   # Number of Acoustic cycles

# Initial Conditions 
LR =  MAT_PROPS["CL"] / EQ_PROPS["FREQ"][0] 
X0 = 300 * 1e-6 /  LR     # Dimensionless Position x/λ_0 (-)

ITERATIONS = 100
NDO        = 1000        # Number of Dense output

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
            solver.set_host(problem_number, "actual_state", 2, 0.0)     # Dimensionless Bubble Position
            solver.set_host(problem_number, "actual_state", 3,  x0)
            solver.set_host(problem_number, "actual_state", 4, 0.0)     # Dimensionless Bubble Wall Velocity
            solver.set_host(problem_number, "actual_state", 5, 0.0)     
            solver.set_host(problem_number, "actual_state", 6, 0.0)     # Dimensionless Bubble Translational Velocity
            solver.set_host(problem_number, "actual_state", 7, 0.0)     

            # Equation properties
            for (k, f) in CP.items():
                # i bubble index (0, 1)
                for i in range(2):
                    solver.set_host(problem_number, "control_parameters", i * SOLVER_OPTS["NCP"] // 2 + k, f(i, **MAT_PROPS, **EQ_PROPS))

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
                          number_of_events=SOLVER_OPTS["NE"],
                          number_of_dense_outputs=NDO,
                          method=SOLVER_OPTS["SOLVER"],
                          threads_per_block=SOLVER_OPTS["BLOCKSIZE"],
                          abs_tol=SOLVER_OPTS["ATOL"],
                          rel_tol=SOLVER_OPTS["RTOL"],
                          event_tol=SOLVER_OPTS["ETOL"],
                          event_dir=SOLVER_OPTS["EDIR"])
    

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
                           x0=X0,
                           td=time_domain)
    

    df = pd.read_csv("KM2B_SOL_DEBUG.csv", names=["t", "r0", "r1", "x0", "x1"])
    plt.figure(1)
    plt.plot(df["t"], (df["x1"]-df["x0"]) * LR * 1e6, "r-", label="D(t)")
    plt.plot(df["t"], (df["r0"]*EQ_PROPS["R0"][0] + df["r1"]*EQ_PROPS["R0"][1])*1e6, "b-", label="$R_0(t)+R_1(t)$")

    solver.syncronize_h2d("all")

    tid = 1
    for ic in range(ITERATIONS):
        print(f"Iteration {ic:.0f}")
        solver.solve_my_ivp()
        solver.syncronize_d2h("actual_state")
        solver.syncronize_d2h("time_domain")
        solver.syncronize_d2h("dense_output")

        t_end = solver.get_host(tid, "time_domain",  0)
        r0    = solver.get_host(tid, "actual_state", 0)
        r1    = solver.get_host(tid, "actual_state", 1)
        x0    = solver.get_host(tid, "actual_state", 2)
        x1    = solver.get_host(tid, "actual_state", 3)
        dense_index, dense_time, dense_states = solver.get_dense_output() 

        plt.plot(t_end, (x1-x0) * LR * 1e6, "r.", markersize=8)
        plt.plot(t_end, (r0*EQ_PROPS["R0"][0] + r1*EQ_PROPS["R0"][1])*1e6, "b.", markersize=8)
        plt.plot(dense_time[:dense_index[tid], tid], (dense_states[:dense_index[tid],3, tid]-dense_states[:dense_index[0],2,tid]) * LR *1e6, 'k.', markersize=2)
        plt.plot(dense_time[:dense_index[tid], tid], (dense_states[:dense_index[tid],0, tid]*EQ_PROPS["R0"][0]+dense_states[:dense_index[0],1,tid]*EQ_PROPS["R0"][1])*1e6, 'k.', markersize=2)

        plt.draw()
        plt.show(block=False)
        input()

