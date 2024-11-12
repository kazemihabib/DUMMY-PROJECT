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

% ================ problem variables ================
int: max_route_length = n + 2;  

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

constraint max_dist = max(k in Couriers)(dist[k]);

solve :: int_search([route[k,p] | k in Couriers, p in Positions], first_fail, indomain_min) 
minimize max_dist;

output [
    "maximum distance of any courier: ", show(fix(max_dist)), "\n",
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
