from .healthcare import load_healthcare_data
from .synthetic import load_synthetic_data

def load_dataset(name):
    if name == "healthcare":
        return load_healthcare_data()
    elif name == "synthetic":
        return load_synthetic_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")

__all__ = ["load_healthcare_data", "load_synthetic_data", "load_dataset"]
