import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.utils.repro import set_global_seed
from src.data.data_loaders import load_polynomial_data, load_nonlinear_data, load_healthcare_data, load_csv_dataset
from src.models.regression_model import RegressionModel
from src.models.twolayer_nn import TwoLayerNNModel
from src.sgd.sgd import SGD
from src.sgd.nonconvex import NonConvexSGD, FixedStepsize, DiminishingStepsize
from src.data.generate_synthetic_data import transform_to_polynomial_train, transform_to_polynomial_test
from src.utils.logging import save_results, make_log_record


# --- at top ---

def run_experiment(dataset, mode, **kwargs):
    seed = kwargs.get("seed", 0)
    save_dir = kwargs.get("save_dir")
    # Load data
    if dataset == "polynomial":
        X_raw, y = load_polynomial_data()
    elif dataset == "nonlinear":
        X_raw, y = load_nonlinear_data()
    elif dataset == "healthcare":
        X_raw, y = load_healthcare_data()
    elif dataset == "csv":
        X_raw, y = load_csv_dataset(
            kwargs["data_path"], 
            target_col=kwargs["target_col"], 
            delimiter=kwargs.get("delimiter", ","),
            header=kwargs.get("header", "infer")
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    raw_dim = X_raw.shape[1]  
    log_records = []

    # ---------------------- CONVEX MODEL ----------------------
    if mode == "convex_model":
        for degree in kwargs.get("degrees", [1, 2, 3, 4, 5]):
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.3, random_state=seed
            )
            X_train_poly, poly, mean, std = transform_to_polynomial_train(X_train, degree=degree)
            X_test_poly = transform_to_polynomial_test(X_test, poly, mean, std)

            for method in kwargs.get("methods", ["fixed", "halving", "diminishing"]):
                model = RegressionModel(X_train_poly, y_train)
                sgd = SGD(model=model, num_epochs=kwargs.get("epochs", 10), batch_size=1, stepsize_type=method)
                w, obj_hist, grad_hist, dist_hist = sgd.optimize()
                X_test_poly_aug = np.hstack([np.ones((X_test_poly.shape[0], 1)), X_test_poly])
                test_loss = np.mean((X_test_poly_aug @ w - y_test) ** 2)

                meta = {
                    "experiment_type": "convex_model_complexity",
                    "model_type": "regression",
                    "degree": degree,
                    "hidden_dim": None,
                    "n": len(X_train),
                    "stepsize_strategy": method,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1,
                    "seed": seed,
                    "raw_dim": raw_dim,
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                print(f"[Convex Model] Degree {degree}, Stepsize {method}, Test Loss: {test_loss:.4f}")

    # ---------------------- CONVEX SAMPLE ----------------------
    elif mode == "convex_sample":
        for degree in kwargs.get("degrees", [1, 2, 3, 4, 5]):
            permuted_idx = np.random.permutation(len(X_raw))  # nested subsets
            for frac in kwargs.get("fracs", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]):
                n = int(len(X_raw) * frac)
                if n <= 3:
                    print(f"Skipping frac={frac} (n={n})")
                    continue
                idx = permuted_idx[:n]
                X_sub, y_sub = X_raw[idx], y[idx]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, y_sub, test_size=0.3, random_state=seed
                )
                X_train_poly, poly, mean, std = transform_to_polynomial_train(X_train, degree=degree)
                X_test_poly = transform_to_polynomial_test(X_test, poly, mean, std)

                for method in kwargs.get("methods", ["fixed", "halving", "diminishing"]):
                    model = RegressionModel(X_train_poly, y_train)
                    sgd = SGD(model=model, num_epochs=kwargs.get("epochs", 10), batch_size=1, stepsize_type=method)
                    w, obj_hist, grad_hist, dist_hist = sgd.optimize()
                    X_test_poly_aug = np.hstack([np.ones((X_test_poly.shape[0], 1)), X_test_poly])
                    test_loss = np.mean((X_test_poly_aug @ w - y_test) ** 2)

                    meta = {
                        "experiment_type": "convex_sample_complexity",
                        "model_type": "regression",
                        "degree": degree,
                        "hidden_dim": None,
                        "n": n,
                        "stepsize_strategy": method,
                        "num_epochs": kwargs.get("epochs", 10),
                        "batch_size": 1,
                        "seed": seed,
                        "raw_dim": raw_dim,
                    }
                    log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                    print(f"[Convex Sample] Degree {degree}, n={n}, Stepsize {method}, Test Loss: {test_loss:.4f}")

    # ---------------------- NONCONVEX MODEL ----------------------
    elif mode == "nonconvex_model":
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=seed)
        degree = kwargs.get("degree", 3)
        X_train_poly, poly, mean, std = transform_to_polynomial_train(X_train, degree=degree)

        poly_model_fixed = RegressionModel(X_train_poly, y_train)
        sgd_fixed_poly = SGD(poly_model_fixed, num_epochs=10, batch_size=1, stepsize_type='fixed')
        sgd_fixed_poly.optimize()
        alpha = sgd_fixed_poly.strategy.alpha

        poly_model_diminish = RegressionModel(X_train_poly, y_train)
        sgd_diminish_poly = SGD(poly_model_diminish, num_epochs=10, batch_size=1, stepsize_type='diminishing')
        sgd_diminish_poly.optimize()
        beta, gamma = sgd_diminish_poly.strategy.beta, sgd_diminish_poly.strategy.gamma

        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=seed)
        for h in kwargs.get("hidden_sizes", [1, 5, 10, 20, 50, 100]):
            for strategy_name, strategy in [("fixed", FixedStepsize(alpha)), ("diminishing", DiminishingStepsize(beta, gamma))]:
                nn = TwoLayerNNModel(input_dim=X_train.shape[1], hidden_dim=h)
                sgd_nn = NonConvexSGD(nn, X_train, y_train, kwargs.get("epochs", 10), 1, strategy)
                w, obj_hist, grad_hist = sgd_nn.optimize()
                test_loss = np.mean((nn.forward_batch(X_test, w) - y_test) ** 2)

                meta = {
                    "experiment_type": "nonconvex_model_complexity",
                    "model_type": "nn",
                    "degree": None,
                    "hidden_dim": h,
                    "n": len(X_train),
                    "stepsize_strategy": strategy_name,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1,
                    "seed": seed,
                    "raw_dim": raw_dim,
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, None, w, test_loss))
                print(f"[Nonconvex Model] Hidden {h}, Stepsize {strategy_name}, Test Loss: {test_loss:.4f}")

    # ---------------------- NONCONVEX SAMPLE ----------------------
    elif mode == "nonconvex_sample":
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X_raw, y, test_size=0.3, random_state=seed)
        degree = kwargs.get("degree", 3)
        X_train_poly, poly, mean, std = transform_to_polynomial_train(X_train0, degree=degree)

        poly_model_fixed = RegressionModel(X_train_poly, y_train0)
        sgd_fixed_poly = SGD(poly_model_fixed, num_epochs=10, batch_size=1, stepsize_type='fixed')
        sgd_fixed_poly.optimize()
        alpha = sgd_fixed_poly.strategy.alpha

        poly_model_diminish = RegressionModel(X_train_poly, y_train0)
        sgd_diminish_poly = SGD(poly_model_diminish, num_epochs=10, batch_size=1, stepsize_type='diminishing')
        sgd_diminish_poly.optimize()
        beta, gamma = sgd_diminish_poly.strategy.beta, sgd_diminish_poly.strategy.gamma

        width = kwargs.get("width", 50)
        for frac in kwargs.get("fracs", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]):
            n = int(len(X_raw) * frac)
            if n <= 3:
                print(f"Skipping frac={frac} (n={n})")
                continue
            idx = np.random.choice(len(X_raw), n, replace=False)
            X_sub, y_sub = X_raw[idx], y[idx]
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.3, random_state=seed)

            for strategy_name, strategy in [("fixed", FixedStepsize(alpha)), ("diminishing", DiminishingStepsize(beta, gamma))]:
                nn = TwoLayerNNModel(input_dim=X_train.shape[1], hidden_dim=width)
                sgd_nn = NonConvexSGD(nn, X_train, y_train, kwargs.get("epochs", 10), 1, strategy)
                w, obj_hist, grad_hist = sgd_nn.optimize()   # <-- 3-tuple
                test_loss = np.mean((nn.forward_batch(X_test, w) - y_test) ** 2)

                meta = {
                    "experiment_type": "nonconvex_sample_complexity",
                    "model_type": "nn",
                    "degree": None,
                    "hidden_dim": width,
                    "n": n,
                    "stepsize_strategy": strategy_name,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1,
                    "seed": seed,
                    "raw_dim": raw_dim,
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, None, w, test_loss))
                print(f"[Nonconvex Sample] n={n}, Stepsize {strategy_name}, Test Loss: {test_loss:.4f}")

    from pathlib import Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = kwargs.get("save_dir", "results") or f"results/{dataset}_{mode}_{timestamp}"
    save_results(log_records, out_path)
    return log_records
    
