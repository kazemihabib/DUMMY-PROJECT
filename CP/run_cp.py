import subprocess
import os
import json
import re
import argparse

def parse_output_into_json(output_string):
    max_distance_match = re.search(r'maximum distance of any courier: (\d+)', output_string)
    max_distance = int(max_distance_match.group(1)) if max_distance_match else None

    max_possible_dist_match = re.search(r'max possible distance: (\d+)', output_string)
    max_possible_dist = int(max_possible_dist_match.group(1)) if max_possible_dist_match else None

    courier_max_lengths = re.search(r'courier max lengths: (\d+)', output_string)
    courier_max_lengths = int(courier_max_lengths.group(1)) if courier_max_lengths else None

    courier_routes = []
    courier_pattern = re.compile(r'courier \d+:\n\s+route: ([\d\s]+)\n')
    for match in courier_pattern.finditer(output_string):
        route = list(map(int, match.group(1).strip().split()))
        courier_routes.append(route)


    time_elapsed_match = re.search(r'% time elapsed: ([\d.]+) s', output_string)
    time_elapsed = int(float(time_elapsed_match.group(1))) if time_elapsed_match else None

    solve_time_match = re.search(r'%%%mzn-stat: solveTime=([\d.]+)', output_string)
    solve_time = float(solve_time_match.group(1)) if solve_time_match else None

    result = {
        "time": time_elapsed,
        # "solve_time": solve_time,
        "optimal": False if time_elapsed > 300 else True,
        "obj": max_distance,
        "sol": courier_routes,
    }    
    return result

def run_subprocess(instance_path, model, solver_type="gecode"):
    try:
        output = subprocess.run(
            [
                "minizinc", 
                "-m", model, 
                "-d", instance_path,
                "--solver", solver_type, 
                "--output-time", 
                "--solver-time-limit", "300000",
                "-s" 
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False, 
            timeout=301,
        )
        if output.stderr and DEBUG:
            print(output.stderr)

        return output.stdout
    
    except:
        print(f"Process timed out and was terminated for model {model}.")
        return None


def solve_instance_cp(instance_path, model, solver_type, output_json):
    output_string = run_subprocess(instance_path, model, solver_type)
    model_name = os.path.basename(model).split(".")[0]
    solved_instance_name = f"{solver_type}_{model_name}"
    if output_string:
        output_json[solved_instance_name] = parse_output_into_json(output_string)
    else: #timeout
        pass
        ...
    
    return output_json

def save_json_output(json_data, instance_file, output_dir):
    json_filename = f"{os.path.splitext(instance_file)[0]}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    ...


def main(instances_dir, output_dir):
    # main section of the code 
    os.makedirs(output_dir, exist_ok=True)

    # solve each instance in the directory
    for instance_file in os.listdir(instances_dir):
        # if os.path.basename(instance_file) in exclude_list: 
        #     continue
        if not instance_file.endswith(".dzn"):
            print("File type must be .dzn.")
            continue
        instance_path = os.path.join(instances_dir, instance_file)
        print(f"Solving {instance_file}")
        
        output_json = {}
        for solver_type, models in solver.items():
            for model in models:
                model = os.path.join(solver_folder, model)
                output_json = solve_instance_cp(instance_path, model, solver_type, output_json)

        if output_json:
            save_json_output(output_json, instance_file, output_dir)


DEBUG = 0
solver_folder = os.path.join(os.path.dirname(__file__),  "solvers")
solver = {
    "gecode":[
        "CP_SYM_LB_RML_HRSTIC_GECODE.mzn",
        "CP_SYM_LB.mzn",
        # "CP.mzn",
    ]
    ,
    "chuffed":[
        "CP_SYM_LB_RML_HRSTIC_CHUFFED.mzn",
        # "CP_SYM_LB.mzn",
        # "CP.mzn",
    ]
}

if __name__=="__main__":
    # instances_dir = "instances/instances_dzn"
    # output_dir = "outputs/test"

    parser = argparse.ArgumentParser(description="run the CP solver.")
    parser.add_argument("instances_dir", help="folder containing instance files.")
    parser.add_argument("output_dir", help="output folder to save results in json.")
    args = parser.parse_args()
    instances_dir = args.instances_dir
    output_dir = args.output_dir

    main(instances_dir, output_dir)

