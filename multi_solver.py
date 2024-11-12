import sys
import os
import SMT.SMT_Z3 as SMT_solver
if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage_message = (
            f"Usage: python {os.path.basename(sys.argv[0])} <input_directory> <output_directory> \n\n"
            "Arguments:\n"
            "  <input_directory>    Directory containing the instance files\n"
            "  <output_directory>   Directory to save the solutions as JSON files\n"
        )
        print(usage_message)
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    # RUN CP    
    print("Running CP")
    try:
        # TODO("change the following line to run the CP solver")
        # os.system(f"python3 SMT/cp.py {input_directory} {output_directory}/CP")
        pass
    except Exception as e:
        print(f"An error occurred while running the CP solver: {e}")

    # Run SMT solver using command line
    print("Running SMT")
    try:
        SMT_solver.solve(input_directory, output_directory, "both")
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


    
    
    
    

