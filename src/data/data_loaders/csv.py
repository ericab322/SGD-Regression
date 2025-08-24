import numpy as np

def load_csv_dataset(path, target_col, delimiter=",", header="infer"):
    try:
        import pandas as pd
        df = pd.read_csv(path, delimiter=delimiter, header=0 if header=="infer" else header)
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
        return X, y
    except ImportError:
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f) if header=="infer" else csv.reader(f, delimiter=delimiter)
            if header != "infer":
                raise RuntimeError("Install pandas for headerless CSVs, or set header='infer'.")
            rows = list(reader)
        y = np.array([float(r[target_col]) for r in rows], dtype=float)
        feature_names = [c for c in rows[0].keys() if c != target_col]
        X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=float)
        return X, y
