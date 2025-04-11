# Theoretically-Grounded SGD for Regression Models

This project explores and implements stochastic gradient descent (SGD) methods for linear, polynomial, and nonlinear regression, with a focus on theoretically-motivated stepsize strategies and convergence analysis. The framework is currently able to support experiments on synthetic data.

## Goals

- Implement a custom, modular SGD framework for regression tasks
- Support linear and polynomial feature expansions with customizable degree
- Evaluate convergence behavior under different stepsize rules:
  - Fixed stepsize (based on theoretical constants)
  - Halving stepsize (based on expected suboptimality gap)
  - Diminishing stepsize
- Generate synthetic datasets under various settings (linear, polynomial, nonlinear, cosine-based on sphere)

## 🛠️ Requirements

- Python 3.8+
- numpy
- scikit-learn
- matplotlib (optional, for plotting)

Install dependencies:
```bash
pip install numpy scikit-learn matplotlib

