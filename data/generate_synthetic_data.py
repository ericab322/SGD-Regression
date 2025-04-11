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
    X = np.random.normal(2, 1, size=(m, n))

    if model_type == 'linear':
        A = np.random.normal(0, 1, size=n)
        b = np.random.normal()
        y = X @ A + b + np.random.normal(0, noise, size=m)
        return X, y, {'A': A, 'b': b}

    elif model_type == 'polynomial':
        A = np.random.normal(0, 1, size=(n,))
        b = np.random.normal()
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        A_full = np.random.normal(0, 1, size=X_poly.shape[1]) 
        y = X_poly @ A_full + b + np.random.normal(0, noise, size=m)
        return X, y, {'A': A_full, 'b': b}

    elif model_type == 'nonlinear':
        b = np.random.normal()
        x = X[:, 0]
        y = nonlinear_func(x) + b + np.random.normal(0, noise, size=m)
        return X, y, {'function': nonlinear_func.__name__, 'b': b}

    else:
        raise ValueError("model_type must be 'linear', 'polynomial', or 'nonlinear'")


def generate_training_data_fixed(m=100, n=2, noise=0.01, degree=2, model_type='linear', nonlinear_func=None):
    """
    Generates data with fixed coefficients for:
        - Linear: y = Ax + b + noise
        - Polynomial: y = A0*x + A1*x^2 + ... + b + noise
        - Nonlinear: y = f(x) + b + noise

    Args:
        m: Number of samples
        n: Number of features (must be 1 for nonlinear)
        noise: Std dev of Gaussian noise
        degree: Degree of polynomial (only used for polynomial)
        model_type: 'linear' | 'polynomial' | 'nonlinear'
        nonlinear_func: function like np.sin, np.exp (only used for nonlinear)
    Returns:
        X: Input data
        y: Output data
        true_coefficients: Dictionary of true coefficients used for generation
    """
    X = np.random.normal(loc=2, scale=1, size=(m, n))
    if model_type == 'linear':
        A = np.array([1.0, 2.0])[:n]  
        b = 1.0                       
        eta_i = np.random.normal(0, noise, size=(m,))
        y = X @ A + b + eta_i
        true_coefficients = {'A': A, 'b': b}
        return X, y, true_coefficients
        
    elif model_type == 'polynomial':
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        d = X_poly.shape[1]

        A = 0.1 * np.arange(1, d + 1)
        b = 1.0  
        y = X_poly @ A + b + np.random.normal(0, noise, size=m)
        return X_poly, y, {'A': A, 'b': b, 'degree': degree}
    
    elif model_type == 'nonlinear':
        b = 1.0
        eta_i = np.random.normal(0, noise, size=(m,))
        y = nonlinear_func(X[:, 0]) + b + eta_i
        true_coefficients = {'function': nonlinear_func.__name__, 'b': b}
        return X, y, true_coefficients
    
    else:
        raise ValueError("model_type must be 'linear', 'polynomial', or 'nonlinear'")
    