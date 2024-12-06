import gurobipy as gp

try:
    # Start Gurobi environment to check the license
    env = gp.Env(empty=True)
    env.start()

except gp.GurobiError as e:
    print(e)
