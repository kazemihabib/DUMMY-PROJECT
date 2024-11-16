from pulp import *
import numpy as np
import os

def load_instance(num_instance):
    """Load instance data from a file and parse it into usable variables."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Instantiate variables from file
    instance_file_path = f"instances/inst{num_instance:02}.dat"

    # Read and parse instance file
    try:
        with open(instance_file_path, 'r') as data_file: #Ensures the file is closed automatically after reading
            lines = [line.strip() for line in data_file.readlines()]
            
        # Extract basic parameters
        num_couriers = int(lines[0])
        num_items = int(lines[1])
        courier_capacity = list(map(int, lines[2].split()))
        item_sizes = list(map(int, lines[3].split()))
        
        # Parse the distance matrix
        distance_matrix = np.array([list(map(int, line.split())) for line in lines[4:]])
        
        return num_couriers, num_items, courier_capacity, item_sizes, distance_matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file '{instance_file_path}' not found.")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing file '{instance_file_path}': {e}")
    
def format_paths(route_decision_vars, num_cities, num_couriers):
    """Format paths for each courier based on the assignment matrix x."""
    solutions = [] # will store the formatted route for each courier.
    
    for courier_id in range(num_couriers):
        courier_path = []

        # Count Assigned Cities
        assigned_cities_count = sum(
            1 for start_city in range(num_cities) 
            for end_city in range(num_cities) 
            if value(route_decision_vars[start_city][end_city][courier_id]) >= 0.9
        )
        
        # Find the starting city for the courier
        start_city = next(
            (city for city in range(num_cities - 1) 
             if value(route_decision_vars[num_cities - 1][city][courier_id]) >= 0.9), 
            None
        )
        
        if start_city is not None:
            courier_path.append(start_city + 1)
            current_city = start_city

            # Trace the path for all assigned cities
            for _ in range(assigned_cities_count - 1):
                next_city = next(
                    (city for city in range(num_cities - 1) 
                     if value(route_decision_vars[current_city][city][courier_id]) > 0.9), 
                    None
                )
                if next_city is not None:
                    courier_path.append(next_city + 1)
                    current_city = next_city
        
        solutions.append(courier_path)
    
    return solutions

def create_solution_json(route_decision_vars, num_cities, num_couriers, solve_time, is_optimal, objective_value):
    """
    Format solution details into a JSON-compatible dictionary.
    
    Parameters:
        x (array): The decision variable matrix.
        num_cities (int): The number of cities in the problem.
        num_couriers (int): The number of couriers in the problem.
        solve_time (float): The time taken to solve the problem.
        is_optimal (bool): Whether the solution is optimal.
        objective_value (float): The objective function value.
        
    Returns:
        dict: A dictionary containing the formatted solution.
    """

    '''No Feasible Solution: The optimization problem could be infeasible.
    For example, if couriers cannot complete the deliveries based on given capacities or time limits,
    the solver might return a negative or default value to signal this infeasibility./ for showing failure'''

    if objective_value < 0:
        return {
            "time": solve_time,
            "optimal": is_optimal,
            "objective": "N/A",
            "solution": []
        }
    
    # Format the solution paths for each courier
    solution_paths = format_paths(route_decision_vars, num_cities, num_couriers)
    
    return {
        "time": solve_time,
        "optimal": is_optimal,
        "objective": round(objective_value),
        "solution": solution_paths
    }

def save_solution_as_json(instance, solution_data):
    """
    Saves the given dictionary as a JSON file in the 'res/MIP' directory under the parent directory.

    Parameters:
        instance (str): The instance identifier to be used as the JSON file name.
        solution_data (dict): The dictionary to save as JSON.

    Returns:
        None
    """
    # Define the target directory and file path
    parent_directory = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_directory, "solution_paths", f"{instance}.json")
    
    # Save the dictionary to a JSON file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, 'w') as file:
        json.dump(solution_data, file, indent=3)
    
    print(f"File saved at {file_path}")
