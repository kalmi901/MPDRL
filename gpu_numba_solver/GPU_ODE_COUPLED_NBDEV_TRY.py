import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPU_ODE_COUPLED
defaults, parameters = GPU_ODE_COUPLED.setup("KM1DNBC", k=2, ac_field="CONST")
from GPU_ODE_COUPLED import CoupledSolverObject

# -- GLOBAl (CONSTANT) PARAMETERS
MAT_PROPS   = defaults["mat_props"].copy()
EQ_PROPS    = defaults["eq_props"].copy()
SOLVER_OPTS = defaults["solver_opts"].copy()


# - Physical Control Parameters -
EQ_PROPS["R0"] = [6.0* 1e-6, 5.0*1e-6, 6.0*1e-6, 5.0*1e-6, 6.0* 1e-6, 5.0*1e-6, 6.0* 1e-6, 5.0*1e-6]
EQ_PROPS["FREQ"][0] = 20.0 * 1e3
EQ_PROPS["FREQ"][1] = 40.0 * 1e3
EQ_PROPS["REL_FREQ"]= 20.0 * 1e3
EQ_PROPS["k"]       = 2

PA0  = [-1.2 * 1.013]                  # Pressure Amplitude 0, (bar)
PA1  = [0.0, 0.1, 0.0, 0.2] * 1
SCALE = "lin"
TIME_DOMAIN = [0.0, 5.0e0]             # Number of Acoustic cycles

# - Initial Positions -
LR = MAT_PROPS["CL"] / EQ_PROPS["FREQ"][0]
X0 = [0, 300, 1000, 1300, 2000, 2300, 3000, 3300]         # Initial bubble positions [micron] (increasing!!!)


# - Solver Configuration -
SOLVER_OPTS["NS"] = len(PA0) * len(PA1)         # Number of Systems
SOLVER_OPTS["UPS"]       = len(EQ_PROPS["R0"])  # Number of Units per System (dual bubble)
SOLVER_OPTS["SPB"]       = 2                    # Systems Per Block
SOLVER_OPTS["BLOCKSIZE"] = SOLVER_OPTS["SPB"] * SOLVER_OPTS["UPS"]   
SOLVER_OPTS["NDO"]       = 512                 # Number of Dense Output


def fill_solver_object(solver,
                       pa0: np.ndarray,
                       pa1: np.ndarray,
                       x0: np.ndarray,
                       td: np.ndarray):
    
    num_bubbles = len(x0)
    system_id = 0
    for p0 in pa0:
        EQ_PROPS["PA"][0] = p0 * 1e5
        for p1 in pa1:
            EQ_PROPS["PA"][1] = p1 * 1e5

            # - Time Domain -
            solver.set_host_value_system_scope(system_id, "time_domain", 0, td[0])
            solver.set_host_value_system_scope(system_id, "time_domain", 1, td[1])

            # - Set Unit Socepe Properties
            for unit_id in range(num_bubbles):
                # - Actual State -
                solver.set_host_value_unit_scope(system_id, unit_id, "actual_state", 0, 1.0)            # R0_i
                solver.set_host_value_unit_scope(system_id, unit_id, "actual_state", 1, x0[unit_id])    # x0_i
                solver.set_host_value_unit_scope(system_id, unit_id, "actual_state", 2, 0.0)            # U0_i
                solver.set_host_value_unit_scope(system_id, unit_id, "actual_state", 3, 0.0)            # V0_i

                # - Unit Parameters -
                for (k, f) in parameters["UP"].items():
                    solver.set_host_value_unit_scope(system_id, unit_id, "unit_parameters", k, f(unit_id, **MAT_PROPS, **EQ_PROPS))


            # - Dynamic parameters - (Acoustic field properties)
            for (k, f) in parameters["DP"].items():
                for i in range(EQ_PROPS["k"]):
                    solver.set_host_value_system_scope(system_id, "dynamic_parameters", i + k*EQ_PROPS["k"], f(i, **MAT_PROPS, **EQ_PROPS))


            # - Coupling Matrix -
            for i in range(num_bubbles):
                for j in range(num_bubbles):
                    if j == i:
                        # - Diagonal (No self coupling) - 
                        solver.set_host_value_coupling_matrix(system_id, 0, i, j, 0.0)
                        solver.set_host_value_coupling_matrix(system_id, 1, i, j, 0.0)
                        solver.set_host_value_coupling_matrix(system_id, 2, i, j, 0.0)
                    else:
                        # - OFF-Diagonals
                        solver.set_host_value_coupling_matrix(system_id, 0, i, j, parameters["CM"][0](i, j, **MAT_PROPS, **EQ_PROPS))
                        solver.set_host_value_coupling_matrix(system_id, 1, i, j, parameters["CM"][1](i, j, **MAT_PROPS, **EQ_PROPS))
                        solver.set_host_value_coupling_matrix(system_id, 2, i, j, parameters["CM"][2](i, j, **MAT_PROPS, **EQ_PROPS))

            system_id += 1

    # GLOBAL PARAMETERES (shared by all systems)
    for (k, f) in parameters["GP"].items():
        solver.set_host_value_global_scope("global_parameters", k, f(**MAT_PROPS, **EQ_PROPS))



if __name__ == "__main__":

    # Create the Solver Object 
    solver_object = CoupledSolverObject(
        number_of_systems=SOLVER_OPTS["NS"],
        number_of_sytems_per_block=SOLVER_OPTS["SPB"],
        number_of_units_per_system=SOLVER_OPTS["UPS"],
        unit_system_dimension=SOLVER_OPTS["UD"],
        number_of_unit_parameters=SOLVER_OPTS["NUP"],
        number_of_system_parameters=SOLVER_OPTS["NSP"],
        number_of_global_parameters=SOLVER_OPTS["NGP"],
        number_of_dynamic_parameters=SOLVER_OPTS["NDP"] * EQ_PROPS["k"],
        number_of_unit_accessories=SOLVER_OPTS["NUA"],
        number_of_system_accessories=SOLVER_OPTS["NSA"],
        number_of_coupling_matrices=SOLVER_OPTS["NC"],
        number_of_coupling_terms=SOLVER_OPTS["NCT"],
        number_of_coupling_factors=SOLVER_OPTS["NCF"],
        number_of_events=SOLVER_OPTS["NE"],
        number_of_dense_outputs=SOLVER_OPTS["NDO"],
        threads_per_block=SOLVER_OPTS["BLOCKSIZE"],
        method=SOLVER_OPTS["SOLVER"],
        linsolve="JACOBI",
        abs_tol=SOLVER_OPTS["ATOL"],
        rel_tol=SOLVER_OPTS["RTOL"],
        event_tol=SOLVER_OPTS["ETOL"],
        event_dir=SOLVER_OPTS["EDIR"]
    )


    time_domain = np.array(TIME_DOMAIN, dtype=np.float64)
    pa0 = np.array(PA0, dtype=np.float64)
    pa1 = np.array(PA1, dtype=np.float64)
    x0 = np.array(X0, dtype=np.float64)  * 1e-6 / LR


    fill_solver_object(solver=solver_object,
                       pa0 = pa0,
                       pa1 = pa1,
                       x0 = x0,
                       td = time_domain)
    
    solver_object.sync_to_device("all")

    solver_object.solve_my_ivp()

    solver_object.sync_to_host("dense_output_index")
    solver_object.sync_to_host("dense_output_time_instances")
    solver_object.sync_to_host("dense_output_states")

    dense_index, dense_time, dense_state = solver_object.get_dense_output()

    print(dense_state[:,0].shape)

    plt.figure(1)


    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 0] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 1] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 2] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 3] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 4] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 5] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 6] * LR * 1e6, '.')
    plt.plot(dense_time[:dense_index[0], 0], dense_state[:dense_index[0], 1, 7] * LR * 1e6, '.')
    plt.show()