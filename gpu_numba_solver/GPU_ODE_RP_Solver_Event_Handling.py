import numpy as np
import pandas as pd
import time
from system_definitions import RP0D
from system_definitions.RP0D import CP, DEFAULT_EQ_PROPS, DEFAULT_MAT_PROPS, DEFAULT_SOLVER_OPTS
import GPU_ODE
GPU_ODE.setup(RP0D)
from GPU_ODE import SolverObject
import matplotlib.pyplot as plt



# GLOBAL (CONSTANT) PARAMETERS
MAT_PROPS   =    DEFAULT_MAT_PROPS.copy()
EQ_PROPS    =    DEFAULT_EQ_PROPS.copy()
SOLVER_OPTS =    DEFAULT_SOLVER_OPTS.copy()

EQ_PROPS["R0"]      = 44.8 * 1e-6
EQ_PROPS["FREQ"]    = 31.25 * 1e3



def fill_solver_object(solver: SolverObject,
                       pa: np.ndarray,
                       freq: np.ndarray,
                       R0: np.float64,
                       time_domain: np.ndarray):
    
    
    EQ_PROPS["R0"] = R0 * 1.0e-6
    problem_number = 0
    for p in pa:
        EQ_PROPS["PA"] = p * 1.0e5
        for f in freq:
            EQ_PROPS["FREQ"] = f * 1.0e3

            # Time Domain
            solver.set_host(problem_number, "time_domain", 0, time_domain[0])
            solver.set_host(problem_number, "time_domain", 1, time_domain[1])

            # Actual State
            solver.set_host(problem_number, "actual_state", 0, 1.0)
            solver.set_host(problem_number, "actual_state", 1, 0.0)

            # Accessories
            solver.set_host(problem_number, "accessories", 0, 1.0)
            solver.set_host(problem_number, "accessories", 1, 0.0)

            for (k, f) in CP.items():
                solver.set_host(problem_number, "control_parameters", k, f(**MAT_PROPS, **EQ_PROPS))

        problem_number += 1


if __name__ == "__main__":

    # Simulation parameters
    MaxIterations = 128
    
    # Sigle parameter set
    R0      = 44.8       # micron
    pa      = np.array([0.225], dtype=np.float64)
    freq    = np.array([31.25], dtype=np.float64)

    # Parameter scan
    # TODO:


    SOLVER_OPTS["NT"] = len(freq) * len(pa)
    time_domain = np.array([0.0, 1.0], dtype=np.float64)

    # Create Solver Object
    solver = SolverObject(number_of_threads=SOLVER_OPTS["NT"],
                          system_dimension=SOLVER_OPTS["SD"],
                          number_of_control_parameters=SOLVER_OPTS["NCP"],
                          number_of_accessories=SOLVER_OPTS["NACC"],
                          number_of_events=SOLVER_OPTS["NE"],
                          method=SOLVER_OPTS["SOLVER"],
                          threads_per_block=SOLVER_OPTS["BLOCKSIZE"],
                          abs_tol=SOLVER_OPTS["ATOL"],
                          rel_tol=SOLVER_OPTS["RTOL"],
                          event_tol=SOLVER_OPTS["ETOL"],
                          event_dir=SOLVER_OPTS["EDIR"])
    
    fill_solver_object(solver, pa, freq, R0, time_domain)
    solver.syncronize_h2d("all")


    df = pd.read_csv("RP_Solution_Debug.csv", names=["t", "r"])

    plt.plot(df["t"], df["r"], "b--", linewidth=1)
    plt.grid("both")

    for ic in range(MaxIterations):
        print(f"Iterations: {ic:.0f}")
        solver.solve_my_ivp()
        solver.syncronize_d2h("accessories")
        solver.syncronize_d2h("actual_state")
        solver.syncronize_d2h("time_domain")

        plt.plot(solver.get_host(0, "accessories", 1), solver.get_host(0, "accessories", 0), "r.")
        plt.plot(solver.get_host(0, "time_domain", 0), solver.get_host(0, "actual_state",0), "k.")
        plt.draw()
        plt.show(block=False)

        input()