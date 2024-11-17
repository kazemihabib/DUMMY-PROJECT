include "globals.mzn";
include "lex_lesseq.mzn";

% ================ problem parameters ================
int: m;  % number of couriers
int: n;  % number of items
int: depot = n+1; % depot must be n+1  

set of int: Couriers = 1..m;
set of int: Items = 1..n;
set of int: Nodes = 1..n+1;  

array[Couriers] of int: l;  % courier capacities
array[Items] of int: s;     % item sizes
array[Nodes, Nodes] of int: D;  % distance matrix

% ================ pre-solve calculations ================
int: total_item = sum(s);
int: max_single_capacity = max(l);
int: LB_dist = 2 * min(i in Items)(D[depot, i]);
% ================ calculating a bound for the route matrix ================
int: max_route_length = ceil(n / m) + 2; % +2 for depot 

float: avg_item_depot_dist = sum(i in Items)(D[i,depot]) / n;% average dist from items to the depot
float: avg_item_dist = sum(i,j in Items where i!=j)(D[i,j]) / (n*(n-1)); % average dist between items (excluding zero distances)
float: UB_dist = (2*avg_item_depot_dist) + ((max_route_length-2)*avg_item_dist); 

% ================ problem variables ================
set of int: Positions = 1..max_route_length;
array[Couriers, Positions] of var 0..n+1: route;
array[Couriers] of var 2..max_route_length: route_length;
array[Couriers] of var int: dist;
array[Couriers] of var 0..total_item: courier_load;
var int: max_dist;

% 1. Routing Constraint: start and end route with depot, limit route_length
constraint forall(k in Couriers)(
    route[k,1] = depot /\
    route[k, route_length[k]] = depot/\
    route_length[k] <= max_route_length 
);

%2. Unused routes set to 0
constraint forall(k in Couriers, p in route_length[k]+1..max_route_length)(
    route[k,p] = 0
);

% 3. Unique Assignment: each item must be assigned exactly once
constraint forall(i in Items)(
    count([route[k,p] | k in Couriers, p in 2..max_route_length-1], i) = 1
);

% 4. Capacity Constraints: calculate courier load and limit it to the capacity
constraint forall(k in Couriers)(
    courier_load[k] = sum(p in 2..route_length[k]-1)(
        if route[k,p] > 0 then s[route[k,p]] else 0 endif
    )
);
constraint forall(k in Couriers)(
    courier_load[k] <= l[k]
);

% 5.Distance Constraint: calculate courier path distance and limit it 
constraint forall(k in Couriers)(
    dist[k] = sum(p in 1..route_length[k]-1)(
        D[route[k,p], route[k,p+1]]
    )
);

% 6. Lower bound Constraint (LB): minimum possible item delivery to ensure each courier delivers something
constraint forall(k in Couriers)(
    dist[k] >= LB_dist
);

constraint max_dist = max(k in Couriers)(dist[k]);

% ================ symmetry-breaking constraint ================
% for couriers of the same capacity 
constraint forall(i,j in Couriers where i<j /\ l[i]=l[j])(
    lex_lesseq([route[i,p] | p in 1..max_route_length], [route[j,p] | p in 1..max_route_length])
);

% search annotation  with multiple phases
solve :: seq_search([
    % 1: route construction with impact-based search
    int_search([route[k,p] | k in Couriers, p in Positions],
               first_fail,
               indomain_split,
               complete),
    % 2: route length optimization
    int_search(route_length, 
               first_fail,
               indomain_random,
               complete),
]) minimize max_dist;

output [
    "maximum distance of any courier: ", show(fix(max_dist)), "\n",
    "\nmax possible distance: ", show(UB_dist), "\n",
    "\nmax route length: ", show(max_route_length), "\n",
    "\nroute length: ", show(route_length), "\n",
    "\ncourier routes and assignments:\n",
    concat([
        let { int: rl = fix(route_length[k]) } in
        concat([
            "courier ", show(k), 
            ":\n    route: ",
            concat([
                show(fix(route[k,p])) ++ " "
                | p in 2..rl-1
            ]),
            "\n    total distance: ", show(fix(dist[k])), "\n"
        ])
        | k in Couriers
    ])
];
