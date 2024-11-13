import sys
import os
import SMT.SMT_Z3 as SMT_solver
import CP.run_cp as CP_solver
if __name__ == "__main__":

    if len(sys.argv) != 2:
        usage_message = (
            f"Usage: python {os.path.basename(sys.argv[0])} <input_directory> <output_directory> \n\n"
            "Arguments:\n"
            "  <output_directory>   Directory to save the solutions as JSON files\n"
        )
        print(usage_message)
        sys.exit(1)
    
    output_directory = sys.argv[1]
    
    # RUN CP    
    print("Running CP")
    try:
        instances_dir = os.path.join(os.path.dirname(__file__), "CP", "instances", "instances_dzn")
        CP_solver.main(instances_dir, f"{output_directory}/CP")
    except Exception as e:
        print(f"An error occurred while running the CP solver: {e}")

    # Run SMT solver using command line
    print("Running SMT")
    try:
        instances_dir = os.path.join(os.path.dirname(__file__), "instances")
        SMT_solver.solve(instances_dir, f"{output_directory}/SMT", "both")
    except Exception as e:
        print(f"An error occurred while running the SMT solver: {e}")

    # Run MIP solver
    print("Running MIP")
    try:
        # TODO("change the following line to run the MIP solver")
        # os.system(f"python3 SMT/mip.py {input_directory} {output_directory}/MIP")
        pass
    except Exception as e:
        print(f"An error occurred while running the MIP solver: {e}")


    
    
    
    

