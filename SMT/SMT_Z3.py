from typing import List, Tuple, Optional
from z3 import Int, Optimize, If, And, Or, Sum, ModelRef, sat, Implies, AtMost, AtLeast, Distinct
from z3 import Function, IntSort
import json
import time
import sys
import os
import re
import math

class MCPSolver:
    """
    MCPSolver is a class that solves the Multiple Couriers Planning (MCP) problem using the Z3 SMT solver.
    Attributes:
        verbose (int): 0 = Quiet/Silent (minimal output) | 1 = Basic information (default level) | 2 = Detailed debug information
        num_couriers (int): The number of couriers.
        num_items (int): The number of items.
        courier_capacities (List[int]): A list of capacities for each courier.
        items_sizes (List[int]): A list of sizes of each item.
        distances_matrix (List[List[int]]): A matrix representing distances between nodes
    """
    def __init__(self, instance_file: str, verbose: int = 1):

        self.num_couriers: int = 0
        self.num_items: int = 0
        self.courier_capacities: List[int] = [] 
        self.items_sizes: List[int]  = [] 
        self.distances_matrix: List[List[int]] = [] 
        self._verbose = verbose
        self._load_instance(instance_file)
        
    def _load_instance(self, filename: str) -> None:
        """
        Load instance from file.
        
        Args:
            filename (str): Path to the instance file

        returns:
            None
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        self.num_couriers = int(lines[0].strip())
        self.num_items = int(lines[1].strip())

        self.courier_capacities = [int(x) for x in lines[2].strip().split()]
        
        self.items_sizes = [int(x) for x in lines[3].strip().split()]
        
        self.distances_matrix = []
        for i in range(self.num_items + 1):
            row = [int(x) for x in lines[4 + i].strip().split()]
            self.distances_matrix.append(row)
            
    def _calculate_max_route_length_of_couriers(self) -> List[int]:
        """ 
        Calculate the maximum possible route length for each courier, 
        decreasing this size improves the performance of the solver. 

        Returns:
            List[int]: Maximum possible route length for each courier
        """
        ## TODO(Maybe we can find a good constraint to limit the max route length for each courier)
        # min_item_size = min(self.items_sizes)
        # couriers_max_route_lengths = [0] * self.num_couriers
        # for courier in range(self.num_couriers):
        #     couriers_max_route_lengths[courier] = min(self.num_items, self.courier_capacities[courier] // min_item_size) + 2
        # return couriers_max_route_lengths       
        return [math.ceil(self.num_items / self.num_couriers) + 2] * self.num_couriers
        
    def _calculate_lower_bound(self) -> int:
        """
        Calculate lower bounds for the objective function.
        
        Returns:
            int: Lower bound value
        """
        depot = self.num_items
        lower_bound = max(self.distances_matrix[depot][i] + self.distances_matrix[i][depot] for i in range(depot))
        
        return lower_bound
    
    def _exactly_one(self, conditions: List[int]):
        """
        Checks whether exactly one of the conditions is true.

        Args:
            args: List of boolean variables 
        """
        return And(AtMost(*conditions, 1), AtLeast(*conditions, 1))

        
    def solve(self, timeout: int = 300, symmetry_breaking = False) -> dict:
        """
        Solve the MCP problem.
        
        Args:
            timeout (int): Maximum solving time in seconds
        
        Returns:
            dict: Solution containing runtime, optimality status, objective value, and routes
        """
        start_time = time.time()
        
        solver = Optimize()
        solver.set("timeout", timeout * 1000)  # Z3 timeout is in milliseconds
        
        # Create matrix of courier assignments
        # My Convention: assignments[j][i] j is the column and i is the row
        assignments = [[Int(f'courier_{i}_at_{j}_node') for j in range(self._calculate_max_route_length_of_couriers()[i])] 
                 for i in range(self.num_couriers)]

        couriers_traveled_distances = [Int(f"courier_{c}_traveled_distance") for c in range(self.num_couriers)]

        # Create objective variable
        max_dist = Int('objective')
        
        self.add_constraints(solver, assignments, couriers_traveled_distances, max_dist, symmetry_breaking)
        
        solver.minimize(max_dist)
        
        status = solver.check()
        end_time = time.time()
        runtime = int(end_time - start_time)
        
        if status == sat:
            model = solver.model()
            solution = self._extract_solution(model, assignments)
            obj_value = model[max_dist].as_long()
            optimal = runtime < timeout
            
            return {
                "time": runtime,
                "optimal": optimal,
                "obj": obj_value,
                "sol": solution
            }
        else:
            return {
                "time": timeout,
                "optimal": False,
                "obj": "",
                "sol": None
            }
            
    def add_constraints(self, solver: Optimize, assignments: List[List], couriers_traveled_distances: List[int], max_dist: int, symmetry_breaking) -> None:
        """
        Adds constraints to the Z3 solver

        Args:
            solver (Optimize): The Z3 solver instance.
            assignments (List[List]): A matrix that solver will assign the items to couriers in that.
            couriers_traveled_distances (List[int]): A list to store the traveled distances of each courier.
            max_dist (int): The objective variable to minimize.
        """
        depot = self.num_items

        # items_sizes_z3 is the function that maps item index to its size
        # it increases the performance of the solver, in defining some constraints
        items_sizes_z3 = Function('items_sizes_z3', IntSort(), IntSort())
        for idx, size in enumerate(self.items_sizes):
            solver.add(items_sizes_z3(idx) == size)

        # distances_matrix_z3 is the function that maps two nodes to the distance between them
        # it increases the performance of the solver, in defining some constraints
        distances_matrix_z3 = Function('distances_matrix_z3', IntSort(), IntSort(), IntSort())
        for i in range(self.num_items + 1):
            for j in range(self.num_items + 1):
                solver.add(distances_matrix_z3(i, j) == self.distances_matrix[i][j])

        couriers_max_route_length = [len(assignments[j]) for j in range(self.num_couriers)]

        # Constraint1: First row must be depot
        # Constraint2: Last row must be depot
        # Constraint3: All other rows must be either depot or an item
        for j in range(self.num_couriers):
            for i in range(couriers_max_route_length[j]):
                # First row must be depot
                # Last row must be depot
                if i == 0 or i == couriers_max_route_length[j] - 1:
                    solver.add(assignments[j][i] == depot)
                else:
                    solver.add(assignments[j][i] >= 0)
                    solver.add(assignments[j][i] <= depot)
        
        # Constraint4: If a courier reaches the depot, it should stay there
        for j in range(self.num_couriers):
            for i in range(1, couriers_max_route_length[j]-1):
                solver.add(Implies(assignments[j][i] == depot, assignments[j][i + 1] == depot)) 

        # Constraint5: the size of items couriers carriers should be <= its capacity
        for j in range(self.num_couriers):
            solver.add(self.courier_capacities[j] >= Sum(
                [If(And(assignments[j][i] < depot, assignments[j][i] >= 0),
                    items_sizes_z3(assignments[j][i]),
                    0)
                for i in range(1, couriers_max_route_length[j] - 1)]  # Skip first and last positions
            ))
        
        # Constraint6: the objective function (max_dist) should be >= to the traveled distance of each courier
        for j in range(self.num_couriers):
            solver.add(couriers_traveled_distances[j] == Sum(
                [distances_matrix_z3(assignments[j][i], assignments[j][i+1])
                for i in range(couriers_max_route_length[j]-1)]
            ))
        
        
        for courier in range(self.num_couriers):
            solver.add(max_dist >= couriers_traveled_distances[courier]) 

        # Constraint7: Each item should be carried by only one courier and all items are delivered by checking
        # all items has been in the assignments matrix only once.
        for node in range(self.num_items):
            flattened = [assignments[j][i] == node  for j in range(self.num_couriers) for i in range(couriers_max_route_length[j]) ]
            solver.add(self._exactly_one(flattened))
        
        # Constraint8: Each courier should deliver at least one item
        for j in range(self.num_couriers):
            solver.add(assignments[j][1] != depot) 
        
        

        # Constraint10: objective function (max_dist) should be greater than the lower bound
        solver.add(max_dist >= self._calculate_lower_bound())

        if symmetry_breaking:
            # Add lexicographical ordering for couriers with the same capacity
            for j1 in range(self.num_couriers-1):
                for j2 in range(j1 + 1, self.num_couriers):
                    if self.courier_capacities[j1] == self.courier_capacities[j2]:
                        # Ensure the first item of courier i is less than that of courier j
                        solver.add(assignments[j1][1] < assignments[j2][1])
                        break # I don't want to say A < B, A < D,  A < F, B < D, B < F, D < F. Instead I want to say A < B, B < D, D < F

        
    def _extract_solution(self, model: ModelRef,
                         assignments: List[List]) -> List[List[int]]:

        """
        Extracts the solution from the given Z3 model.
        This method processes the Z3 model to extract the routes for each courier,
        excluding depot visits. It returns a list of routes where each route is a 
        list of item indices (Starting from 1 not zero).

        Args:
            model (ModelRef): The Z3 model containing the solution.
            assignments (List[List]): The matrix representing the problem variables.

        Returns:
            List[List[int]]: A list of routes for each courier, where each route is 
            a list of item indices (1-based).
        """

        if self._verbose == 2:
            for i in range(max(len(assignments[j]) for j in range(self.num_couriers))):
                for j in range(self.num_couriers):
                    if ( i < len(assignments[j])):
                        number = model[assignments[j][i]].as_long()
                        print(str(number).zfill(2), end="  ")
                print()

        solution: List[List[int]] = [[] for _ in range(self.num_couriers)]
        
        for j in range(self.num_couriers):
            # Get route excluding depot visits
            route = []
            for i in range(1, len(assignments[j])-1):
                val = model[assignments[j][i]].as_long()
                if val < self.num_items:  # If not depot
                    route.append(val + 1) # +1 to start from 1 instead of zero
            solution[j] = route
            
        return solution
        
    def save_solution(self, solution: dict, solution_name: str, output_file: str) -> None:

        data = {}
    
        # Try to load existing file if it exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If file exists but isn't valid JSON, start fresh
                pass
                
        # Add new solution with provided name
        data[solution_name] = solution
        
        # Write updated data back to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

def solve(input_dir: str, output_dir: str, symmetry_breaking: str, verbose:int = 1) -> None:
    """
    Solves the problem instances located in the input directory and saves the solutions in the output directory.
    Parameters:
    input_dir (str): The directory containing the input files with '.dat' extension.
    output_dir (str): The directory where the output files will be saved.
    symmetry_breaking (str): Specifies whether to use symmetry breaking. 
                             Acceptable values are "both", "sb" (with symmetry breaking), and "nosb" (without symmetry breaking).
    verbose (int, optional): The verbosity level.
        0 = Quiet/Silent (minimal output)
        1 = Basic information (default level)
        2 = Detailed debug information

    Returns: None
    """
    os.makedirs(output_dir, exist_ok=True)
     
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.dat'):
            input_file = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0]
            output_filename_number_part = re.search(r'\d+', output_filename).group()
            output_filename = output_filename_number_part + '.json'
            output_file = os.path.join(output_dir, output_filename)
            
            solver = MCPSolver(input_file, verbose=verbose)
            if symmetry_breaking in ["both", "sb"]:
                print(f"Solving {filename} with symmetry breaking")
                solution = solver.solve(timeout = 300, symmetry_breaking = True)
                if verbose == 1:
                    print(solution)
                solver.save_solution(solution, "Z3_SB", output_file)
            if symmetry_breaking in ["both", "nosb"]:
                print(f"Solving {filename} without symmetry breaking")
                solution = solver.solve(timeout = 300, symmetry_breaking = False)
                if verbose == 1:
                    print(solution)
                solver.save_solution(solution, "Z3_NO_SB", output_file)

if __name__ == "__main__":
    verbose = True

    if len(sys.argv) != 4:
        usage_message = (
            f"Usage: python {os.path.basename(sys.argv[0])} <input_directory> <output_directory> <sb|nosb>\n\n"
            "Arguments:\n"
            "  <input_directory>    Directory containing the instance files\n"
            "  <output_directory>   Directory to save the solutions as JSON files\n"
            "  <both|sb|nosb>            Symmetry Breaking option\n"
            "                       both:   Run both with and without symmetry breaking\n"
            "                       sb    : Enable symmetry breaking\n"
            "                       nosb  : Disable symmetry breaking\n"
        )
        print(usage_message)
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)
    if sys.argv[3] not in ["sb", "nosb", "both"]: 
        print("Symmetry breaking option should be one of sb, nosb, both")
        sys.exit(1)

    symmetry_breaking = sys.argv[3]

    solve(input_dir, output_dir, symmetry_breaking)
        