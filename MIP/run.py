import sys
import os
## Temporary solution to fix No module named 'MIP' error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pulp import *
from MIP.helper import *
from MIP.MIP_model import *
from math import floor,log
import time # Import time to track execution
import multiprocessing
import gurobipy 
from gurobipy import GRB
import argparse

TIME_LIMIT = 300

def solve_model(output_dir, model, instance, solver_name, solver, solution_data, route_decision_vars, num_couriers, num_items, time_limit = TIME_LIMIT):
    """
    Solves the given optimization model with the specified solver and places the results in a queue.
    """
   
    try :
        
        model.solve(solver)
        status = model.status
        if status == 1:  # Feasible solution
            solve_time = min(time_limit, floor(model.solutionTime))
            is_optimal = solve_time < time_limit
            solution_data[solver_name] = create_solution_json(
                route_decision_vars, num_items + 1, num_couriers, solve_time, is_optimal, value(model.objective)
            )
        else:
            solution_data[solver_name] = create_solution_json(None, 0, 0, time_limit, False, -1)
    except:
        solution_data[solver_name] = create_solution_json(None, 0, 0, time_limit, False, -1)
        
        save_solution_as_json(instance,solution_data, output_dir)




def run(input_dir, output_dir, instance=0, time_limit=TIME_LIMIT):
    """
    Executes the optimization process for one or multiple instances using various solvers
    with forced termination if solving exceeds the time limit.
    """

    
    solvers = {
        "CBC": PULP_CBC_CMD(timeLimit=time_limit),
        "HiGHS": getSolver('HiGHS', timeLimit=time_limit, msg=False),
        "Gurobi": GUROBI(timeLimit=time_limit),
    }

    if instance == 0:
        first_instance = 1
        last_instance = 22
    else:
        first_instance = instance
        last_instance = instance + 1

    for instance in range(first_instance, last_instance):
        # Load instance data and set up the model
        num_couriers, num_items, courier_capacity, item_sizes, distance_matrix = load_instance(instance, input_dir)
        model, route_decision_vars, courier_distances = setup_model(
            num_couriers, num_items, courier_capacity, item_sizes, distance_matrix
        )

        solution_data = {}
        
    for solver_name, solver in solvers.items():
        process = multiprocessing.Process(target=solve_model, args=(output_dir, model, instance, solver_name, solver, solution_data, route_decision_vars, num_couriers, num_items))
        process.start()  # Start the solver in a separate process
        process.join(timeout = time_limit)  # Wait for the process to complete or timeout

        if process.is_alive():
            # Solver exceeded the time limit
            process.terminate()  # Force terminate the solver
            process.join()  # Ensure the process is cleaned up


if __name__=="__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="run the MIP solver.")
    parser.add_argument("instances_dir", help="folder containing instance files.")
    parser.add_argument("output_dir", help="output folder to save results in json.")

    # Parse the command-line arguments
    args = parser.parse_args()
    instances_dir = args.instances_dir
    output_dir = args.output_dir
    # get abs path
    instances_dir = os.path.abspath(instances_dir)
    output_dir = os.path.abspath(output_dir)

    # Execute the main function
    run(instances_dir, output_dir)







    
