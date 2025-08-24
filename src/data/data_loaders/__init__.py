from src.data.data_loaders.nonlinear import load_nonlinear_data
from .healthcare import load_healthcare_data
from .polynomial import load_polynomial_data
from .nonlinear import load_nonlinear_data
from .csv import load_csv_dataset

def load_dataset(name):
    if name == "healthcare":
        return load_healthcare_data()
    elif name == "synthetic":
        return load_polynomial_data()
    elif name == "nonlinear":
        return load_nonlinear_data()
    elif name == "csv":
        return load_csv_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

__all__ = ["load_healthcare_data", "load_polynomial_data", "load_nonlinear_data", "load_dataset"]
