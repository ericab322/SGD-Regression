# Theoretically-Grounded SGD Framework

This project explores and implements stochastic gradient descent (SGD) methods for convex and nonconvex objectives.

## Features

- Modular SGD framework for both convex (e.g., regression) and nonconvex (e.g., neural networks) objectives
- Supports:
  - Linear regression
  - Polynomial regression (with configurable degree)
  - Two-layer fully connected neural networks
- Tracks convergence behavior using:
  - **Fixed stepsize**
  - **Diminishing stepsize**
  - **Halving stepsize**
- Automatically estimates and logs theoretical constants:
  - `L` (Lipschitz smoothness)
  - `c` (strong convexity)
  - `mu`, `M`, `M_G` (gradient-related bounds)
- Compatible with both synthetic and real-world datasets (e.g., CSV input)
- Separate logs for:
  - Experiment results (`experiment_log.csv`)
  - Optimization parameters (`sgd_parameters_log.csv`)

---

## Setting Up the Environment
This project requires specific dependencies to ensure compatibility and reproducibility. These dependencies are listed in the environment.yaml file. Follow the steps below to set up the environment:

1. **Locate the `environment.yaml` file**  
   The YAML file specifies the environment name and the required dependencies for this project.

2. **Create the environment using Conda**  
   Run the following command to create the environment:

   ```bash
   conda env create -f environment.yaml
3. **Activate the Environment**
    Once the environment is created, activate it using the following command:

    ```bash
    conda activate sgd-regression
    
