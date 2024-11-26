from pulp import *
import numpy as np
import os

def load_instance(num_instance, input_dir):
    """
    Load instance data from a file and parse it into usable variables.

    This function reads an instance file containing problem data for a logistics or optimization task.
    The file is expected to include the number of couriers, number of items, courier capacities, item sizes, 
    and a distance matrix. These parameters are parsed and returned for use in solving the problem.

    Parameters:
        num_instance (int): The instance number (e.g., 1, 2, ...) used to identify the input file.
        input_dir (str): The directory where instance files are stored.

    Returns:
        tuple: A tuple containing the following elements:
            - num_couriers (int): The total number of couriers.
            - num_items (int): The total number of items.
            - courier_capacity (list): A list of integers representing each courier's capacity.
            - item_sizes (list): A list of integers representing the size of each item.
            - distance_matrix (np.ndarray): A 2D numpy array representing distances between locations.

    Raises:
        FileNotFoundError: If the specified instance file does not exist.
        ValueError: If the file contents are incorrectly formatted or missing necessary data.
    """
    # Get the directory of the current script and set it as the working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Construct the full file path for the instance file
    instance_file_path = os.path.join(input_dir, f"inst{num_instance:02}.dat")

    # Read and parse the instance file
    try:
        with open(instance_file_path, 'r') as data_file:  # Ensures the file is closed automatically
            # Strip leading and trailing whitespace from each line in the file
            lines = [line.strip() for line in data_file.readlines()]
        
        # Extract parameters from the file
        num_couriers = int(lines[0])  # The first line specifies the number of couriers
        num_items = int(lines[1])  # The second line specifies the number of items
        courier_capacity = list(map(int, lines[2].split()))  # Third line: courier capacities
        item_sizes = list(map(int, lines[3].split()))  # Fourth line: item sizes

        # Parse the distance matrix from the remaining lines
        distance_matrix = np.array([list(map(int, line.split())) for line in lines[4:]])

        # Return the parsed data
        return num_couriers, num_items, courier_capacity, item_sizes, distance_matrix
    
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise FileNotFoundError(f"Instance file '{instance_file_path}' not found.")
    except (ValueError, IndexError) as e:
        # Raise an error if the file format is incorrect
        raise ValueError(f"Error parsing file '{instance_file_path}': {e}")
    
def format_paths(route_decision_vars, num_cities, num_couriers):
    """
    Formats paths for each courier based on the assignment matrix.

    This function processes the decision variables from a Mixed-Integer Programming (MIP) solution 
    to determine the routes assigned to each courier. It identifies the sequence of cities visited 
    by each courier and returns the formatted routes.

    Parameters:
        route_decision_vars (list): A 3D list (or matrix) where route_decision_vars[i][j][k] represents 
            whether courier k travels from city i to city j (binary decision variable).
        num_cities (int): The total number of cities, including the depot as the last city (city num_cities - 1).
        num_couriers (int): The total number of couriers.

    Returns:
        list: A list of lists, where each inner list contains the sequence of city indices (1-based) 
              representing the route for each courier. If no route is assigned, the courier's list will be empty.
    """
    solutions = []  # Stores the formatted routes for each courier.
    
    for courier_id in range(num_couriers):
        courier_path = []  # Initialize an empty route for the current courier.

        # Count the number of cities assigned to this courier.
        assigned_cities_count = sum(
            1 for start_city in range(num_cities) 
            for end_city in range(num_cities) 
            if value(route_decision_vars[start_city][end_city][courier_id]) >= 0.9
        )
        
        # Find the starting city for this courier.
        start_city = next(
            (city for city in range(num_cities - 1)  # Exclude the depot.
             if value(route_decision_vars[num_cities - 1][city][courier_id]) >= 0.9), 
            None
        )
        
        if start_city is not None:
            # Add the starting city to the route (convert to 1-based index).
            courier_path.append(start_city + 1)
            current_city = start_city

            # Trace the path for all assigned cities except the starting city.
            for _ in range(assigned_cities_count - 1):
                next_city = next(
                    (city for city in range(num_cities - 1)  # Exclude the depot.
                     if value(route_decision_vars[current_city][city][courier_id]) > 0.9), 
                    None
                )
                if next_city is not None:
                    courier_path.append(next_city + 1)  # Convert to 1-based index.
                    current_city = next_city
        
        # Append the courier's route to the solutions list.
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
            "obj": "N/A",
            "sol": []
        }
    
    # Format the solution paths for each courier
    solution_paths = format_paths(route_decision_vars, num_cities, num_couriers)
    
    return {
        "time": solve_time,
        "optimal": is_optimal,
        "obj": round(objective_value),
        "sol": solution_paths
    }

def save_solution_as_json(instance, solution_data : dict, solver_name, output_dir: str) :

    file_path = os.path.join(output_dir, f"{instance}.json")

    data = {}
    
    # Try to load existing file if it exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but isn't valid JSON, start fresh
            pass
                
    # Add new solution with provided name
    data[solver_name] = solution_data

    # Write updated data back to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=3)


