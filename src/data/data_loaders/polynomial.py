import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from src.data.generate_synthetic_data import generate_training_data_fixed, transform_to_polynomial_train

def load_polynomial_data(m=102799, n=6, degree=3, noise=0.001, normalize=True):
    """
    Loads synthetic data for experiments.
    Returns:
        X_raw: untransformed features
        X_poly: polynomial features of given degree
        y: target values
    """
    X_raw, _, _ = generate_training_data_fixed(m=m, n=n)
    X_poly, poly, mean, std = transform_to_polynomial_train(X_raw, degree=degree,)

    true_A = 0.1 * np.arange(1, X_poly.shape[1] + 1)
    true_b = 1.0


    y = X_poly @ true_A + true_b + noise * np.random.randn(len(X_raw))


    return X_raw, y
