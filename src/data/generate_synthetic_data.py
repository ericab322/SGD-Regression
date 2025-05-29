import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def generate_cosine_data(m=100, d=3, noise=0.01):
    """
    Generates data on the sphere in R^d with labels y = cos(<x, u>) + noise.
    Args:
        m: Number of samples
        d: Dimension of the sphere
        noise: Std dev of Gaussian noise

    Returns:
        X: data points on the sphere
        y: target values using cosine of dot product
        u: the true vector
    """
    X = np.random.randn(m, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  

    u = np.random.randn(d)
    u /= np.linalg.norm(u)

    y = np.cos(X @ u) + np.random.normal(0, noise, size=m)
    
    return X, y, u


def generate_training_data_unfixed(m=100, n=2, noise=0.01, model_type='linear', degree=2, nonlinear_func=None):
    """
    Generates synthetic training data with random A, b:
        - Linear: y = Ax + b + noise
        - Polynomial: y = A^T phi(x) + noise, where phi is polynomial basis
        - Nonlinear: y = f(x_1) + b + noise
     Args:
        m: Number of samples
        n: Number of features (must be 1 for nonlinear)
        noise: Std dev of Gaussian noise
        model_type: 'linear' | 'polynomial' | 'nonlinear'
        degree: Degree of polynomial (only used for polynomial)
        nonlinear_func: function like np.sin, np.exp (only used for nonlinear)

    Returns:
        X: Input data 
        y: Target values
        true_coefficients: Dictionary of true parameters
    """
    X = np.random.normal(0, 1, size=(m, n))
    A = np.random.normal(0, 1, size=n)
    b = np.random.normal()
    y = X @ A + b + np.random.normal(0, noise, size=m)
    return X, None, y, {'A': A, 'b': b}


def generate_training_data_fixed(m=100, n=2, noise=0.01):
    """
    Generates data with fixed coefficients for:
        - Linear: y = Ax + b + noise
        - Polynomial: y = A0*x + A1*x^2 + ... + b + noise
        - Nonlinear: y = f(x) + b + noise

    Args:
        m: Number of samples
        n: Number of features (must be 1 for nonlinear)
        noise: Std dev of Gaussian noise
    Returns:
        X: Input data
        y: Output data
        true_coefficients: Dictionary of true coefficients used for generation
    """
    X = np.random.normal(loc=0, scale=1, size=(m, n))
    A = 0.01 * np.arange(1, n + 1)
    b = 1.0                       
    eta_i = np.random.normal(0, noise, size=(m,))
    y = X @ A + b + eta_i
    true_coefficients = {'A': A, 'b': b}
    return X, y, true_coefficients

def transform_to_polynomial(X, degree=2, normalize=True):
    """
    Transforms input data to polynomial features of a given degree.
    Args:
        X: Input data
        degree: Degree of polynomial features
        normalize: If True, normalize the features
    Returns:
        X_poly: Transformed polynomial features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    if normalize:
        X_poly = (X_poly - np.mean(X_poly, axis=0)) / np.std(X_poly, axis=0)
    return X_poly
    
def transform_to_nonlinear(X, func=np.sin):
    """
    Transforms input data to nonlinear features using a given function.
    Args:
        X: Input data
        func: Nonlinear function to apply
    Returns
        X_nonlinear: Transformed nonlinear features
    """
    return func(X)

def generate_sphere_data(m, d, degree=3, noise=0.01):
    X = np.random.randn(m, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  
    X_poly = transform_to_polynomial(X, degree=degree, normalize=False)

    A = 0.01 * np.arange(1, X_poly.shape[1] + 1)
    b = 1.0
    y = X_poly @ A + b + noise * np.random.randn(m)

    return X, y, A, b
    