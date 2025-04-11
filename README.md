# Theoretically-Grounded SGD for Regression Models

This project explores and implements stochastic gradient descent (SGD) methods for linear, polynomial, and nonlinear regression, with a focus on theoretically-motivated stepsize strategies and convergence analysis. The framework is currently able to support experiments on synthetic data.

## Project Goals

- Implement a custom, modular SGD framework for regression tasks
- Support linear and polynomial feature expansions with customizable degree
- Evaluate convergence behavior under different stepsize rules:
  - **Fixed stepsize** 
  - **Halving stepsize**
  - **Diminishing stepsize**
- Generate synthetic datasets under various settings:
  - Linear regression
  - Polynomial regression
  - Nonlinear functions (e.g., `sin(x)`, `exp(x)`)
  - Cosine labels on the unit sphere

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
4. **Activate the Environment**
  Once the environment is created, activate it using the following command:
conda activate multiclass-classification
