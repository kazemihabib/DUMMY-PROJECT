from pulp import *
from .helper import *
from math import floor,log
import numpy as np

TIME_LIMIT = 300

def setup_model(num_couriers, num_items, courier_capacity, item_size, distance_matrix):
    """
    x[i][j][c]: A binary decision variable indicating whether courier c travels from city i to city j.
    u[i][c]: Integer decision variable for the visit order of courier c to city i and Used for sub-tour elimination constraints in TSP problems.
    """

    model = LpProblem("Minimize_m_TSP",LpMinimize)

    num_cities = distance_matrix.shape[0]-1
    depot_index = num_cities+1
    

    lower_bound, nearest_city_dist = calculate_lower_bound(distance_matrix, num_couriers, num_items)
    upper_bound = calculate_upper_bound(num_couriers, num_items, distance_matrix)
   
    x = LpVariable.dicts("x", (range(depot_index), range(depot_index), range(num_couriers)), cat="Binary")
    u = LpVariable.dicts("u", (range(num_cities), range(num_couriers)), lowBound=0, upBound = depot_index - 1, cat="Integer")
    max_distance = LpVariable("max_dist",lowBound = lower_bound, upBound = upper_bound, cat="Integer")
    courier_weigths = [LpVariable(name=f'weigth_{i}', lowBound=0, upBound = courier_capacity[i], cat="Integer")
                   for i in range(num_couriers)]
    courier_distance = [
            LpVariable(name=f'obj_dist{i}', cat="Integer", lowBound = nearest_city_dist, upBound = upper_bound)
            for i in range(num_couriers)]
    
    model += max_distance

    # Apply the constraints
    set_courier_weights(model, courier_weigths, x, item_size, num_items, num_couriers)
    ensure_no_useless_arcs(model, x, depot_index, num_couriers)
    ensure_single_visit_to_city(model, x, num_cities, depot_index, num_couriers)
    ensure_departure_from_depot(model, x, num_cities, num_couriers)
    ensure_return_to_depot(model, x, num_cities, num_couriers)
    ensure_connected_paths(model, x, depot_index, num_couriers)
    eliminate_subroutes(model, x, u, num_cities, num_couriers)
    set_courier_distance(model, courier_distance, x, distance_matrix, depot_index, num_couriers)
    set_maximum_distance(model, courier_distance, max_distance)

    return model,x,courier_distance


def calculate_upper_bound(num_couriers, num_items, distance_matrix):
    """
    num_items : Total number of items to deliver.
    max_round_trip_distance : The longest round trip between the depot and a city.
    Divide by num_couriers to distribute the workload evenly across couriers.
    """
    num_cities = num_items
    depot_index = num_items + 1

    # # Compute round-trip distances using NumPy
    round_trip_distances = distance_matrix[num_cities, :num_cities] + distance_matrix[:num_cities, num_cities]

    lower_bound = np.max(round_trip_distances)
    max_round_trip_distance = np.max(round_trip_distances)
    upper_bound = ( (num_items // num_couriers) * round(max_round_trip_distance) ) + lower_bound

    return upper_bound

def calculate_lower_bound(distance_matrix, num_couriers, num_items):
    """
    Calculates a lower bound for the total travel distance.
    
    D[n_cities, :n_cities]: Distances from the depot to each city.
    D[:n_cities, n_cities]: Distances from each city back to the depot.
    + : to compute the round-trip distance for each city.

    """
    '''the goal is to minimize the maximum workload or travel distance among all agents.
    The maximum round-trip distance is used as a starting point to ensure all agents can cover their assigned routes, even in the worst-case scenario.'''
    
    num_cities = num_items
    depot_index = num_items + 1

    # Compute round-trip distances using NumPy
    round_trip_distances = distance_matrix[num_cities, :num_cities] + distance_matrix[:num_cities, num_cities]

    # Find minimum round-trip distances
    nearest_city_dist = np.min(round_trip_distances)
    lower_bound = np.max(round_trip_distances)  # each courier at least goes to one city and returns
    
    return lower_bound, nearest_city_dist

# Set Weight Carried by Each Courier
def set_courier_weights(model, courier_weigths, x, item_size, num_items, num_couriers):
    for c in range(num_couriers):
        terms = [
            x[i][j][c] * item_size[j]
            for i in range(num_items + 1)
            for j in range(num_items)
        ]
        model += courier_weigths[c] == lpSum(terms)

def ensure_no_useless_arcs(model, x, depot_index, num_couriers):
    useless_arcs = sum(x[i][i][c] for i in range(depot_index) for c in range(num_couriers))
    model += useless_arcs == 0

def ensure_single_visit_to_city(model, x, num_cities, depot_index, num_couriers):
    # Loop over all cities to ensure each city is visited exactly once
    for j in range(num_cities):
        visit_sum = lpSum(x[i][j][c] for i in range(depot_index) for c in range(num_couriers))  # Sum of the visit variables for city j
        model += visit_sum == 1  # Ensure the city j is visited exactly once by any courier

def ensure_departure_from_depot(model, x, num_cities, num_couriers):
    for c in range(num_couriers):
        # Summing up the departure decisions for each courier (courier c leaving from the depot)
        departure_sum = lpSum(x[num_cities][j][c] for j in range(num_cities))
        model += departure_sum == 1  # Ensure that each courier leaves the depot exactly once

def ensure_return_to_depot(model, x, num_cities, num_couriers):
    for c in range(num_couriers):
        # Summing up the return decisions for each courier (courier c returning to the depot)
        return_sum = lpSum(x[i][num_cities][c] for i in range(num_cities))
        model += return_sum == 1  # Ensure that each courier returns to the depot exactly once

def ensure_connected_paths(model, x, depot_index, num_couriers):
    for j in range(depot_index):
        for c in range(num_couriers):
            # Sum of all outbound arcs from city j for courier c
            outgoing = lpSum(x[i][j][c] for i in range(depot_index))  
            
            # Sum of all inbound arcs to city j for courier c
            incoming = lpSum(x[j][i][c] for i in range(depot_index))  
            
            # Ensure the number of outbound arcs equals the number of inbound arcs for each city and courier
            model += outgoing == incoming

def eliminate_subroutes(model, x, u, num_cities, num_couriers):
    for c in range(num_couriers):
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    # Define the constraint terms more explicitly for clarity
                    u_diff = u[i][c] - u[j][c]
                    distance_term = num_cities * x[i][j][c]
                    # Add the constraint to the model
                    model += u_diff + distance_term <= num_cities - 1

def set_courier_distance(model, courier_distance, x, distance_matrix, depot_index, num_couriers):
    for c in range(num_couriers):
        # Create a list of distance terms for the courier c
        distance_terms = [
            x[i][j][c] * distance_matrix[i][j] for i in range(depot_index) for j in range(depot_index)
        ]
        
        # Add the constraint to the model
        model += lpSum(distance_terms) == courier_distance[c]

def set_maximum_distance(model, courier_distance, max_distance):
    # Add constraints that the max distance should be greater than or equal to each courier's distance
    for distance in courier_distance:
        model += max_distance >= distance

