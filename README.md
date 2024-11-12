
# CDMO

## Prerequisites

- **Docker**: Ensure Docker is installed on your host machine. You can download it from [Docker's official website](https://www.docker.com/get-started).
- **Python**: Version 3.6 or higher (Recommended: Python 3.10 as we tested with this version)
- **pip**: Python package installer

## Building docker image

- Open your favorite terminal
- Navigate to the root of this project
- Run the following command

        docker build -t cdmo .
    This will create an image called `cdmo`

## Running docker image

Simply run the following command, it will run the `cdmo` image and will give you a bash shell to run the 
solvers explained in next steps.

    docker run -it --rm cdmo

**Recommended**: If you want to have access the generated soloutions by solvers in your host, mount a directory in your
host into the docker using the following command:

    docker run -it --rm -v <host_res_directory>:<docker_res_directory> cdmo

Examples:

    docker run -it --rm -v $(pwd)/docker_generated_files/res:/app/res cdmo

    docker run -it --rm -v ~/docker_generated_files/res:/app/res cdmo

Now you have a prompt to run solvers

## Activate the virtual environment

Run the following command

        source venv/bin/activate

## Running every solver

run the following docker 

           python SMT_Z3.py <input_directory> <output_directory>

            Arguments:
            input_directory    Directory containing the instance files
            output_directory   Directory to save the solutions as JSON files

it will all instances in the input directory and save the results in the output directory

Example:

    python3 multi_solver.py Instances/ res



## Running the SMT Solver

```bash
python SMT/SMT_Z3.py <input_dir> <output_dir> <sb|nosb>
```

### Arguments

- `<input_directory>`: Directory containing the instance files
- `<output_directory>`: Directory to save the solutions as JSON files
- `<sb|nosb|both>`: Symmetry Breaking option
    
    - `both`: Run both with and without symmetry breaking
    - `sb`: Enable symmetry breaking
    - `nosb`: Disable symmetry breaking

## Example

To run the SMT Solver with symmetry breaking enabled:
```bash
python3 SMT/SMT_Z3.py Instances res/SMT sb
```

To run the SMT Solver without symmetry breaking:
```bash
python3 SMT/SMT_Z3.py Instances res/SMT nosb
```

To run the SMT Solver with and without symmetry breaking:
```bash
python3 SMT/SMT_Z3.py Instances res/SMT both 
```
