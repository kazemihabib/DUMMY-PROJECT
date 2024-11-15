# CDMO

This project tackles the **Multiple Couriers Planning (MCP) problem**, a complex combinatorial optimization challenge where a set of couriers must deliver items to various locations while minimizing the maximum distance any courier travels. The MCP problem, increasingly relevant in the era of online shopping and rapid delivery expectations, requires that each courier’s load capacity is respected, and routes are planned efficiently from an origin point back to the same point.

The goal of this project is to model and solve the MCP problem using three approaches:
1. **Constraint Programming (CP)**
2. **Satisfiability Modulo Theories (SMT)**
3. **Mixed-Integer Linear Programming (MIP)**

Each model applies different optimization techniques to allocate items fairly and efficiently to couriers. The project includes experiments to evaluate solver performance across multiple instances.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start Guide (TL;DR)](#quick-start-guide-tldr)
- [Detailed Setup and Usage](#detailed-setup-and-usage)
  - [Building Docker Image](#building-docker-image)
  - [Running Docker Image](#running-docker-image)
  - [Running Solvers](#running-solvers)
    - [Running All Solvers](#running-all-solvers)
    - [Running the SMT Solver](#running-the-smt-solver)
    - [Running the CP Solver](#running-the-cp-solver)
    - [Running the MIP Solver](#running-the-mip-solver)

## Prerequisites

Before you begin, ensure that you have the following installed:

- **Docker**: Install Docker on your host machine. You can download it from [Docker's official website](https://www.docker.com/get-started).
- **Python**: Version 3.6 or higher is required. We recommend using Python 3.10, as this is the version we tested.
- **pip**: Python's package installer.


---

## Quick Start Guide (TL;DR)

1. **Open your terminal and navigate to the root of this project**.

2. **Build Docker Image**:  

    ```bash
    docker build -t cdmo .
    ```

3. **Run Docker**:

    ```bash
    docker run -it --rm cdmo
    ```

    This command will launch the Docker container and open a shell session inside it, allowing you to run commands directly within the container environment.

4. **Run All Solvers**:

    ```bash
    python multi_solver.py res/
    ```
5. **Check correctness of found soloutions**
    ```bash
    python check_solution.py instances/ res/
    ```
6. **Access soloutions**
    ```bash
    cd /app/res
    ```

For more specific instructions on setting up and running individual solvers or using additional options, see [Detailed Setup and Usage](#detailed-setup-and-usage)

## Detailed Setup and Usage





### Building Docker Image

To build the Docker image:

1. Open your preferred terminal.
2. Navigate to the root of this project.
3. Run the following command:

   ```bash
   docker build -t cdmo .
   ```

   This will create an image named `cdmo`.

### Running Docker Image

To run the Docker image, execute the following command:

```bash
docker run -it --rm cdmo
```

This command will run the `cdmo` image and provide a bash shell for running the solvers.

**Optional1**: If you want to access the solution files generated by the solvers from your host machine, mount a directory from your host into the Docker container by adding `-v <host_res_directory>:<docker_res_directory>` option to the previous command

**Examples**

```bash
docker run -it --rm -v $(pwd)/docker_generated_files/res:/app/res cdmo

docker run -it --rm -v ~/docker_generated_files/res:/app/res cdmo
```

**Optional2**: If you have new instance files in your host that you want to solve, Again mount the new instances directory from your host into the Docker by adding `-v <host_new_instances_directory>:<docker_new_input_directory>`

**Examples**

```bash
docker run -it --rm -v $(pwd)/new_instances:/app/new_instances cdmo

docker run -it --rm -v ~/new_instances:/app/new_instances cdmo
```

You can use both *Optional1* and *Optional2* together

**Example**
```bash
docker run -it --rm -v $(pwd)/docker_generated_files/res:/app/res -v $(pwd)/new_instances:/app/new_instances cdmo
```


Now, you have a prompt ready to run the solvers.

### Running Solvers

#### Running All Solvers

To run all solvers:

```bash
python multi_solver.py <output_directory>
```

- `<output_directory>`: Directory where the solutions will be saved as JSON files.

This will solve all instances given in the assignment using *CP*, *SMT* and *MIP* and save the results in the output directory.

**Example**:

```bash
python3 multi_solver.py instances/ res/
```
#### Running the SMT Solver

To run the SMT solver:

```bash
python SMT/SMT_Z3.py <input_directory> <output_directory> <sb|nosb|both>
```

Arguments:
- `<input_directory>`: Directory containing the instance files.
- `<output_directory>`: Directory to save the solutions as JSON files.
- `<sb|nosb|both>`: Symmetry breaking options:
  - `both`: Run both with and without symmetry breaking.
  - `sb`: Enable symmetry breaking.
  - `nosb`: Disable symmetry breaking.

**Examples**

To run the SMT Solver with symmetry breaking enabled:

```bash
python3 SMT/SMT_Z3.py instances res/SMT sb
```

To run the SMT Solver without symmetry breaking:

```bash
python3 SMT/SMT_Z3.py instances res/SMT nosb
```

To run the SMT Solver with and without symmetry breaking:

```bash
python3 SMT/SMT_Z3.py instances res/SMT both
```
#### Running the CP Solver

To run the CP solver:

```bash
python CP/run_cp.py <input_directory> <output_directory>
```

Arguments:
- `<input_directory>`: Directory containing the instance files.
- `<output_directory>`: Directory to save the solutions as JSON files.

The script will run three different minizinc solvers using each "chuffed" or "gecode". The solvers include:
- `CP_SYM_LB_RML_HRSTIC`: Uses Symmetry-breaking, Lower-bound constraint, Route Matrix Limiting and non-trivial Heuristics.
- `CP_SYM_LB`: Uses only Symmetry-breaking and Lower-bound constraint.
- `CP`: The simplest solver without Symmetry-breaking or LB constraint. 


#### Running the MIP solver

To run the MIP solver:

```bash
python run.py <input_directory> <output_directory>
```

Arguments:
- `<input_directory>`: Directory containing the instance files.
- `<output_directory>`: Directory to save the solutions as JSON files.

Solvers:
The script supports multiple solvers, including:

- `CBC` : Default solver provided by PuLP.
- `HiGHS` : High-performance solver for linear programming.
