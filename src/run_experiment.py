import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.data_loaders import load_synthetic_data, load_healthcare_data
from src.models.regression_model import RegressionModel
from src.models.twolayer_nn import TwoLayerNNModel
from src.sgd.sgd import SGD
from src.sgd.nonconvex import NonConvexSGD, FixedStepsize, DiminishingStepsize
from src.data.generate_synthetic_data import transform_to_polynomial
from src.utils.logging import save_results, make_log_record

np.random.seed(0)


def run_experiment(dataset, mode, **kwargs):
    # Load data 
    if dataset == "synthetic":
        X_raw, X_poly, y = load_synthetic_data()
    elif dataset == "healthcare":
        X, y = load_healthcare_data()
        X_raw, X_poly = X, transform_to_polynomial(X, degree=kwargs.get("degree", 3), normalize=True)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    log_records = []

    # Convex model complexity
    if mode == "convex_model":
        for degree in kwargs.get("degrees", [1, 2, 3, 4, 5]):
            X_poly_deg = transform_to_polynomial(X_raw, degree=degree, normalize=True)
            X_train, X_test, y_train, y_test = train_test_split(X_poly_deg, y, test_size=0.3)

            for method in kwargs.get("methods", ["fixed", "halving", "diminishing"]):
                model = RegressionModel(X_train, y_train)
                sgd = SGD(model=model, num_epochs=kwargs.get("epochs", 5), batch_size=1, stepsize_type=method)
                w, obj_hist, grad_hist, dist_hist = sgd.optimize()
                test_loss = np.mean((X_test @ w[1:] + w[0] - y_test) ** 2)

                meta = {
                    "experiment_type": "convex_model_complexity",
                    "model_type": "regression",
                    "degree": degree,
                    "hidden_dim": None,
                    "n": len(X_train),
                    "stepsize_strategy": method,
                    "num_epochs": kwargs.get("epochs", 5),
                    "batch_size": 1
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                print(f"Completed convex model complexity for degree {degree} with {method} stepsize")

    # Convex sample complexity
    elif mode == "convex_sample":
        degree = kwargs.get("degree", 2)
        X_poly_full = transform_to_polynomial(X_raw, degree=degree, normalize=True)
        for frac in kwargs.get("fracs", [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0]):
            n = int(len(X_poly_full) * frac)
            X_train, X_test, y_train, y_test = train_test_split(X_poly_full[:n], y[:n], test_size=0.3)

            for method in kwargs.get("methods", ["fixed", "halving", "diminishing"]):
                model = RegressionModel(X_train, y_train)
                sgd = SGD(model=model, num_epochs=kwargs.get("epochs", 10), batch_size=1, stepsize_type=method)
                w, obj_hist, grad_hist, dist_hist = sgd.optimize()
                test_loss = np.mean((X_test @ w[1:] + w[0] - y_test) ** 2)

                meta = {
                    "experiment_type": "convex_sample_complexity",
                    "model_type": "regression",
                    "degree": degree,
                    "hidden_dim": None,
                    "n": n,
                    "stepsize_strategy": method,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                print(f"Completed convex sample complexity for fraction {frac} with {method} stepsize")

    # Nonconvex model complexity 
    elif mode == "nonconvex_model":
        # alpha, beta, gamma from regression
        poly_model_fixed = RegressionModel(X_poly, y)
        sgd_fixed_poly = SGD(poly_model_fixed, num_epochs=5, batch_size=1, stepsize_type='fixed')
        sgd_fixed_poly.optimize()
        alpha = sgd_fixed_poly.strategy.alpha

        poly_model_diminish = RegressionModel(X_poly, y)
        sgd_diminish_poly = SGD(poly_model_diminish, num_epochs=5, batch_size=1, stepsize_type='diminishing')
        sgd_diminish_poly.optimize()
        beta, gamma = sgd_diminish_poly.strategy.beta, sgd_diminish_poly.strategy.gamma

        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3)
        for h in kwargs.get("hidden_sizes", [1, 5, 10, 20, 50, 100]):
            for strategy_name, strategy in [
                ("fixed", FixedStepsize(alpha)),
                ("diminishing", DiminishingStepsize(beta, gamma))
            ]:
                nn = TwoLayerNNModel(input_dim=X_train.shape[1], hidden_dim=h)
                sgd_nn = NonConvexSGD(nn, X_train, y_train, kwargs.get("epochs", 10), 1, strategy)
                w, obj_hist, grad_hist, dist_hist = sgd_nn.optimize()
                test_loss = np.mean((nn.forward_batch(X_test, w) - y_test) ** 2)

                meta = {
                    "experiment_type": "nonconvex_model_complexity",
                    "model_type": "nn",
                    "degree": None,
                    "hidden_dim": h,
                    "n": len(X_train),
                    "stepsize_strategy": strategy_name,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                print(f"Completed nonconvex model complexity for hidden size {h} with {strategy_name} stepsize")

    # Nonconvex sample complexity
    elif mode == "nonconvex_sample":
        # Get alpha, beta, gamma from regression
        poly_model_fixed = RegressionModel(X_poly, y)
        sgd_fixed_poly = SGD(poly_model_fixed, num_epochs=5, batch_size=1, stepsize_type='fixed')
        sgd_fixed_poly.optimize()
        alpha = sgd_fixed_poly.strategy.alpha

        poly_model_diminish = RegressionModel(X_poly, y)
        sgd_diminish_poly = SGD(poly_model_diminish, num_epochs=5, batch_size=1, stepsize_type='diminishing')
        sgd_diminish_poly.optimize()
        beta, gamma = sgd_diminish_poly.strategy.beta, sgd_diminish_poly.strategy.gamma

        width = kwargs.get("width", 50)
        for frac in kwargs.get("fracs", [0.05, 0.1, 0.2, 0.5, 1.0]):
            n = int(len(X_raw) * frac)
            X_sub, y_sub = X_raw[:n], y[:n]
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.3)

            for strategy_name, strategy in [
                ("fixed", FixedStepsize(alpha)),
                ("diminishing", DiminishingStepsize(beta, gamma))
            ]:
                nn = TwoLayerNNModel(input_dim=X_train.shape[1], hidden_dim=width)
                sgd_nn = NonConvexSGD(nn, X_train, y_train, kwargs.get("epochs", 10), 1, strategy)
                w, obj_hist, grad_hist, dist_hist = sgd_nn.optimize()
                test_loss = np.mean((nn.forward_batch(X_test, w) - y_test) ** 2)

                meta = {
                    "experiment_type": "nonconvex_sample_complexity",
                    "model_type": "nn",
                    "degree": None,
                    "hidden_dim": width,
                    "n": n,
                    "stepsize_strategy": strategy_name,
                    "num_epochs": kwargs.get("epochs", 10),
                    "batch_size": 1
                }
                log_records.append(make_log_record(meta, obj_hist, grad_hist, dist_hist, w, test_loss))
                print(f"Completed nonconvex sample complexity for fraction {frac} with {strategy_name} stepsize")

    # Save results
    save_results(log_records, f"results/{dataset}_{mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "healthcare"], required=True)
    parser.add_argument("--mode", choices=["convex_model", "convex_sample", "nonconvex_model", "nonconvex_sample"], required=True)
    args = parser.parse_args()

    run_experiment(args.dataset, args.mode)
    print(f"Experiment completed for dataset: {args.dataset}, mode: {args.mode}")