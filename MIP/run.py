from pulp import *
from helper import *
from MIP_model import *
from math import floor,log
import time
import gurobipy 
from gurobipy import GRB

TIME_LIMIT = 300

def run(instance, time_limit = TIME_LIMIT):

    solvers = {
                "CBC":PULP_CBC_CMD(timeLimit=time_limit),
                "HiGHS":getSolver('HiGHS', timeLimit=time_limit,msg=False),
                "Gurobi": GUROBI(timeLimit=time_limit)
            }
    
    # Load instance data and define model constraints
    num_couriers, num_items, courier_capacity, item_sizes, distance_matrix = load_instance(instance)
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
    save_solution_as_json(instance,solution_data)


if __name__ == "__main__":
    instance = 7
    run(instance)

