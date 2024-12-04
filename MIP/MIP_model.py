from pulp import *
from MIP.helper import *
from math import floor,log
import numpy as np

TIME_LIMIT = 300

def setup_model(num_couriers, num_items, courier_capacity, item_size, distance_matrix):
    """
    Sets up the optimization model for a multiple Traveling Salesman Problem (m-TSP) with capacity constraints.

    This function defines the decision variables, objective function, and constraints required to solve the m-TSP 
    problem. Couriers must transport items between cities while minimizing the maximum distance traveled by any courier, 
    subject to constraints like capacity limits and sub-tour elimination.

    Parameters:
        num_couriers (int): The number of couriers available.
        num_items (int): The total number of items to be transported.
        courier_capacity (list): A list of integers representing the capacity of each courier.
        item_size (list): A list of integers representing the size of each item.
        distance_matrix (numpy.ndarray): A 2D array representing distances between cities and the depot.

    Returns:
        tuple: A tuple containing the following:
            - model (LpProblem): The optimization model.
            - x (dict): A dictionary of binary decision variables where:
                `x[i][j][c]` indicates whether courier `c` travels from city `i` to city `j`.
            - courier_distance (list): A list of decision variables representing the total distance traveled by each courier.
    
    Decision Variables:
        - x[i][j][c]: Binary variable indicating whether courier `c` travels from city `i` to city `j`.
        - u[i][c]: Integer variable representing the visit order of courier `c` at city `i`.
        - max_distance: Integer variable representing the maximum distance traveled by any courier.
        - courier_weights: Integer variables representing the total weight of items carried by each courier.
        - courier_distance: Integer variables representing the total distance traveled by each courier.

    Constraints:
        - Weight limits: Ensures no courier exceeds their capacity.
        - Path structure: Ensures each courier starts and ends at the depot, visiting each city exactly once.
        - Sub-tour elimination: Prevents sub-routes within the city visits.
        - Distance calculation: Computes the total distance for each courier and ensures the objective function reflects the maximum.

    Objective:
        - Minimize the maximum distance traveled by any courier.

    Steps:
        1. Define the lower and upper bounds for distances using helper functions.
        2. Create decision variables for the optimization problem.
        3. Define constraints for courier weights, arcs, and paths.
        4. Eliminate sub-tours using the `u` variables.
        5. Add the distance computation and the objective function to the model.

    Returns:
        model, x, courier_distance: The formulated optimization problem and decision variables.
    """
    # Create the optimization model
    model = LpProblem("Minimize_m_TSP", LpMinimize)

    num_cities = distance_matrix.shape[0] - 1  # Exclude the depot
    depot_index = num_cities + 1  # Index of the depot

    # Compute round-trip distances from the depot to each city
    round_trip_distances = distance_matrix[num_cities, :num_cities] + distance_matrix[:num_cities, num_cities]
    # Compute bounds for distances
    lower_bound, nearest_city_dist = calculate_lower_bound(round_trip_distances, distance_matrix, num_couriers, num_items)
    upper_bound = calculate_upper_bound(round_trip_distances, num_couriers, num_items, distance_matrix)

    # Define decision variables
    x = LpVariable.dicts("x", (range(depot_index), range(depot_index), range(num_couriers)), cat="Binary")
    u = LpVariable.dicts("u", (range(num_cities), range(num_couriers)), lowBound=0, upBound=depot_index - 1, cat="Integer")
    max_distance = LpVariable("max_dist", lowBound=lower_bound, upBound=upper_bound, cat="Integer")
    courier_weights = [
        LpVariable(name=f'weight_{i}', lowBound=0, upBound=courier_capacity[i], cat="Integer")
        for i in range(num_couriers)
    ]
    courier_distance = [
        LpVariable(name=f'obj_dist{i}', cat="Integer", lowBound=nearest_city_dist, upBound=upper_bound)
        for i in range(num_couriers)
    ]

    # Set the objective function to minimize the maximum distance
    model += max_distance

    # Apply constraints
    set_courier_weights(model, courier_weights, x, item_size, num_items, num_couriers)
    ensure_no_useless_arcs(model, x, depot_index, num_couriers)
    ensure_single_visit_to_city(model, x, num_cities, depot_index, num_couriers)
    ensure_departure_from_depot(model, x, num_cities, num_couriers)
    ensure_return_to_depot(model, x, num_cities, num_couriers)
    ensure_connected_paths(model, x, depot_index, num_couriers)
    eliminate_subroutes(model, x, u, num_cities, num_couriers)
    set_courier_distance(model, courier_distance, x, distance_matrix, depot_index, num_couriers)
    set_maximum_distance(model, courier_distance, max_distance)

    return model, x, courier_distance

def calculate_upper_bound(round_trip_distances, num_couriers, num_items, distance_matrix):
    """
    Calculates an upper bound for the maximum distance a courier might travel.

    This function estimates an upper bound on the maximum distance traveled by any courier, 
    considering the total number of items to deliver and the distribution of workload among couriers. 
    It uses the round-trip distances from the depot to cities to approximate the workload.

    Parameters:
        num_couriers (int): The number of couriers available for deliveries.
        num_items (int): The total number of items to deliver, corresponding to the number of cities (excluding the depot).
        distance_matrix (numpy.ndarray): A 2D array representing distances between all cities and the depot.

    Returns:
        int: An upper bound on the maximum distance any courier might travel.

    Steps:
        1. Compute the round-trip distances between the depot and each city.
        2. Determine the longest round-trip distance to set a lower bound.
        3. Estimate the upper bound by distributing the workload (items) evenly across couriers.

    Variables:
        - `num_cities`: The number of cities, equivalent to the number of items.
        - `depot_index`: The index of the depot in the distance matrix.
        - `round_trip_distances`: An array of distances for round trips between the depot and each city.
        - `lower_bound`: The longest single round-trip distance from the depot to a city.
        - `max_round_trip_distance`: The longest round-trip distance for any city.
        - `upper_bound`: The estimated maximum distance a courier might travel.

    Assumptions:
        - Workload (number of items) is evenly distributed among the couriers.
        - Couriers start and return to the depot.

    Example:
        Given 3 couriers, 5 items, and a 6x6 distance matrix:
        - Each courier handles approximately `5 / 3` items.
        - The upper bound is calculated based on the longest round trip and the total workload.
    """
    # Calculate the longest round-trip distance (lower bound)
    lower_bound = np.max(round_trip_distances)
    max_round_trip_distance = np.max(round_trip_distances)

    # Calculate the upper bound by distributing workload evenly across couriers
    upper_bound = ((num_items // num_couriers) * round(max_round_trip_distance)) + lower_bound

    return upper_bound

def calculate_lower_bound(round_trip_distances, distance_matrix, num_couriers, num_items):
    """
    Calculates a lower bound for the total travel distance and the minimum distance 
    from the depot to the nearest city.

    This function computes the minimum and maximum round-trip distances between 
    the depot and cities. The lower bound ensures that all couriers can handle 
    their routes even under the worst-case scenario, where each courier travels 
    the longest round-trip distance.

    Parameters:
        distance_matrix (numpy.ndarray): A 2D array representing distances between all cities and the depot.
        num_couriers (int): The number of couriers available for deliveries.
        num_items (int): The total number of items to deliver, corresponding to the number of cities (excluding the depot).

    Returns:
        tuple: 
            - lower_bound (int): The maximum round-trip distance between the depot and any city, 
              ensuring that each courier can handle at least one trip.
            - nearest_city_dist (int): The minimum round-trip distance between the depot and the closest city.

    Steps:
        1. Compute the round-trip distances for all cities:
            - Sum the distance from the depot to a city and back to the depot.
        2. Calculate the nearest city's round-trip distance as the minimum.
        3. Determine the lower bound as the longest round-trip distance.

    Variables:
        - `num_cities`: The number of cities, equivalent to the number of items.
        - `depot_index`: The index of the depot in the distance matrix.
        - `round_trip_distances`: An array of distances for round trips between the depot and each city.
        - `nearest_city_dist`: The shortest round-trip distance from the depot to a city.
        - `lower_bound`: The longest round-trip distance, ensuring that each courier can handle at least one city.

    Assumptions:
        - Each courier visits at least one city and returns to the depot.
        - All couriers' routes start and end at the depot.

    Example:
        Given 3 couriers, 5 items, and a 6x6 distance matrix:
        - The lower bound ensures every courier can handle the worst-case travel distance.
        - The nearest city distance is used as a reference for constraints.

    """
    # Find the nearest city's round-trip distance
    nearest_city_dist = np.min(round_trip_distances)

    # The lower bound is the longest round-trip distance
    lower_bound = np.max(round_trip_distances)

    return lower_bound, nearest_city_dist

# Set Weight Carried by Each Courier
def set_courier_weights(model, courier_weigths, x, item_size, num_items, num_couriers):
    for c in range(num_couriers):
        # Define weight carried by each courier as sum of item sizes in their route
        terms = [
            x[i][j][c] * item_size[j]
            for i in range(num_items + 1)
            for j in range(num_items)
        ]
        model += courier_weigths[c] == lpSum(terms)

# Ensure no useless arcs (courier traveling from a city to itself)
def ensure_no_useless_arcs(model, x, depot_index, num_couriers):
    useless_arcs = sum(x[i][i][c] for i in range(depot_index) for c in range(num_couriers))
    model += useless_arcs == 0

# Ensure each city is visited exactly once
def ensure_single_visit_to_city(model, x, num_cities, depot_index, num_couriers):
    for j in range(num_cities):
        # Sum of visit variables for city j
        visit_sum = lpSum(x[i][j][c] for i in range(depot_index) for c in range(num_couriers))
        model += visit_sum == 1  # Ensure the city is visited exactly once

# Ensure each courier departs from the depot exactly once
def ensure_departure_from_depot(model, x, num_cities, num_couriers):
    for c in range(num_couriers):
        # Sum of departure decisions for each courier from the depot
        departure_sum = lpSum(x[num_cities][j][c] for j in range(num_cities))
        model += departure_sum == 1  # Ensure the courier leaves the depot once

# Ensure each courier returns to the depot exactly once
def ensure_return_to_depot(model, x, num_cities, num_couriers):
    for c in range(num_couriers):
        # Sum of return decisions for each courier to the depot
        return_sum = lpSum(x[i][num_cities][c] for i in range(num_cities))
        model += return_sum == 1  # Ensure the courier returns to the depot once

# Ensure that each city has equal inbound and outbound arcs for each courier (connected paths)
def ensure_connected_paths(model, x, depot_index, num_couriers):
    for j in range(depot_index):
        for c in range(num_couriers):
            # Sum of all outbound arcs from city j for courier c
            outgoing = lpSum(x[i][j][c] for i in range(depot_index))  
            
            # Sum of all inbound arcs to city j for courier c
            incoming = lpSum(x[j][i][c] for i in range(depot_index))  
            
            # Ensure outbound arcs equal inbound arcs for each city and courier
            model += outgoing == incoming

# Eliminate subroutes by enforcing sub-tour elimination constraints
def eliminate_subroutes(model, x, u, num_cities, num_couriers):
    for c in range(num_couriers):
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    # Sub-tour elimination constraint for each courier and city pair
                    u_diff = u[i][c] - u[j][c]
                    distance_term = num_cities * x[i][j][c]
                    model += u_diff + distance_term <= num_cities - 1

# Set the total travel distance for each courier
def set_courier_distance(model, courier_distance, x, distance_matrix, depot_index, num_couriers):
    for c in range(num_couriers):
        # List of distance terms for courier c based on the decision variables and distance matrix
        distance_terms = [
            x[i][j][c] * distance_matrix[i][j] for i in range(depot_index) for j in range(depot_index)
        ]
        
        # Add constraint that the total distance equals the courier's distance variable
        model += lpSum(distance_terms) == courier_distance[c]

# Set the maximum distance constraint for the model
def set_maximum_distance(model, courier_distance, max_distance):
    # Ensure that the max distance is greater than or equal to each courier's distance
    for distance in courier_distance:
        model += max_distance >= distance

        