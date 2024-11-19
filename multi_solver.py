import sys
import os
import SMT.SMT_Z3 as SMT_solver
import CP.run_cp as CP_solver
from MIP import run
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
        # raise Exception("CP solver is skipped")
        instances_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "instances", "instances_dzn"))
        cp_output_dir = os.path.abspath(os.path.join(output_directory, "CP"))
        os.makedirs(cp_output_dir, exist_ok=True)
        CP_solver.main(instances_dir, cp_output_dir)
    except Exception as e:
        print(f"An error occurred while running the CP solver: {e}")

    # Run SMT solver using command line
    print("Running SMT")
    try:
        # raise Exception("SMT solver is skipped")
        instances_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "instances", "instances_dat"))
        smt_output_dir = os.path.abspath(os.path.join(output_directory, "SMT"))
        os.makedirs(smt_output_dir, exist_ok=True)
        SMT_solver.solve(instances_dir, smt_output_dir, "both")
    except Exception as e:
        print(f"An error occurred while running the SMT solver: {e}")

    # Run MIP solver
    print("Running MIP")
    try:
        # raise Exception("MIP solver is skipped")
        instances_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "instances", "instances_dat"))
        mips_output_dir = os.path.abspath(os.path.join(output_directory, "MIP"))
        os.makedirs(mips_output_dir, exist_ok=True)
        run.run(instances_dir, mips_output_dir)
    except Exception as e:
        print(f"An error occurred while running the MIP solver: {e}")


    
    
    
    

