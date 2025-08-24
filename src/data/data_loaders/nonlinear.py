import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from src.data.generate_synthetic_data import generate_training_data_fixed, transform_to_polynomial_train
from src.data.generate_synthetic_data import make_nonlinear_dataset

def load_nonlinear_data(m=102799, n=6, degree=3, noise=0.001):
    """
    Loads synthetic data for experiments.
    Returns:
        X_raw: untransformed features
        X_poly: polynomial features of given degree
        y: target values
    """
    X, y = make_nonlinear_dataset(n=n, d=n, noise_std=noise, seed=0)
    return X, y
