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

def run(input_dir, output_dir, instance = 0, time_limit = TIME_LIMIT):

    solvers = {
                "CBC":PULP_CBC_CMD(timeLimit=time_limit),
                "HiGHS":getSolver('HiGHS', timeLimit=time_limit,msg=False),
                "Gurobi": GUROBI(timeLimit=time_limit)
            }
    
    if instance == 0 :
        first_instance = 1
        last_instance = 22
    else :
        first_instance = instance
        last_instance = instance + 1

    for instance in range(first_instance, last_instance) :

        # Load instance data and define model constraints
        num_couriers, num_items, courier_capacity, item_sizes, distance_matrix = load_instance(instance, input_dir)
        model, route_decision_vars, courier_distances = setup_model(num_couriers, num_items, courier_capacity, item_sizes, distance_matrix)  

        solution_data = {}
        
        for solver in solvers:
            model.solve(solvers[solver])
            status = model.status
            if status == 1:    
                solve_time = min(time_limit, floor(model.solutionTime))
                is_optimal = solve_time < time_limit
                solution_data[solver] = create_solution_json(route_decision_vars,num_items + 1,num_couriers,solve_time,is_optimal,value(model.objective))
            else:
                solution_data[solver] = create_solution_json(route_decision_vars,num_items + 1,num_couriers,300,False,-1)
        save_solution_as_json(instance,solution_data, output_dir)


if __name__=="__main__":
    # instances_dir = "instances/instances_dzn"
    # output_dir = "outputs/test"

    parser = argparse.ArgumentParser(description="run the MIP solver.")
    parser.add_argument("instances_dir", help="folder containing instance files.")
    parser.add_argument("output_dir", help="output folder to save results in json.")
    args = parser.parse_args()
    instances_dir = args.instances_dir
    output_dir = args.output_dir
    # get abs path
    instances_dir = os.path.abspath(instances_dir)
    output_dir = os.path.abspath(output_dir)

    run(instances_dir, output_dir)







    
