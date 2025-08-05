import numpy as np
from src.data.generate_synthetic_data import generate_training_data_fixed, transform_to_polynomial

def load_synthetic_data(m=5000, n=4, degree=3, noise=0.001, normalize=True):
    """
    Loads synthetic data for experiments.
    Returns:
        X_raw: untransformed features
        X_poly: polynomial features of given degree
        y: target values
    """
    X_raw, _, _ = generate_training_data_fixed(m=m, n=n, noise=noise)

    X_poly = transform_to_polynomial(X_raw, degree=degree, normalize=normalize)

    true_A = 0.1 * np.arange(1, X_poly.shape[1] + 1)
    true_b = 1.0
    y = X_poly @ true_A + true_b + noise * np.random.randn(X_poly.shape[0])

    return X_raw, X_poly, y
