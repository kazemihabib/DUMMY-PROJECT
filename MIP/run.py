import sys
import os
## Temporary solution to fix No module named 'MIP' error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pulp import *
from MIP.helper import *
from MIP.MIP_model import *
from math import floor,log
import time
import gurobipy 
from gurobipy import GRB
import argparse

TIME_LIMIT = 300

def run(input_dir, output_dir, instance=0, time_limit=TIME_LIMIT):
    """
    Executes the optimization process for one or multiple instances using various solvers.

    This function loads instance data, sets up the optimization model, and solves it using multiple solvers. 
    It saves the solution data in JSON format for each instance, detailing the results for each solver.

    Parameters:
        input_dir (str): The directory where the input instance files are stored.
        output_dir (str): The directory where the solution files will be saved.
        instance (int): The instance number to run. If set to 0, the function will process all instances from 1 to 21.
        time_limit (int): The maximum time (in seconds) allowed for each solver to find a solution.

    Returns:
        None
    """
    # Define the solvers with their respective time limits
    solvers = {
        "CBC": PULP_CBC_CMD(timeLimit=time_limit),  # CBC solver
        "HiGHS": getSolver('HiGHS', timeLimit=time_limit, msg=False),  # HiGHS solver
        "Gurobi": GUROBI(timeLimit=time_limit)  # Gurobi solver
    }

    # Determine the range of instances to process
    if instance == 0:  # If instance is 0, process all instances
        first_instance = 1
        last_instance = 22  # Process instances from 1 to 21
    else:  # Otherwise, process only the specified instance
        first_instance = instance
        last_instance = instance + 1

    # Loop through each instance in the specified range
    for instance in range(first_instance, last_instance):
        # Load instance data and setup the optimization model
        num_couriers, num_items, courier_capacity, item_sizes, distance_matrix = load_instance(instance, input_dir)
        model, route_decision_vars, courier_distances = setup_model(
            num_couriers, num_items, courier_capacity, item_sizes, distance_matrix
        )

        solution_data = {}  # Store solution results for all solvers

        # Solve the model using each solver
        for solver in solvers:
            model.solve(solvers[solver])  # Solve the model
            status = model.status  # Check the solver status

            if status == 1:  # If the solution is feasible
                solve_time = min(time_limit, floor(model.solutionTime))  # Record the solution time
                is_optimal = solve_time < time_limit  # Check if the solution is optimal
                solution_data[solver] = create_solution_json(
                    route_decision_vars, num_items + 1, num_couriers, solve_time, is_optimal, value(model.objective)
                )
            else:  # If no feasible solution is found
                solution_data[solver] = create_solution_json(
                    route_decision_vars, num_items + 1, num_couriers, 300, False, -1
                )

        # Save the solution data for the current instance
        save_solution_as_json(instance, solution_data, output_dir)

if __name__=="__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="run the MIP solver.")
    parser.add_argument("instances_dir", help="folder containing instance files.")
    parser.add_argument("output_dir", help="output folder to save results in json.")

    # Parse the command-line arguments
    args = parser.parse_args()
    instances_dir = args.instances_dir
    output_dir = args.output_dir

    # Execute the main function
    run(instances_dir, output_dir)







    
