import numpy as np
import pandas as pd
import os

def run_experiment_and_log(
    model_type,
    X_train, y_train,
    X_test, y_test,
    stepsize_strategy='fixed',
    num_iterations=1000,
    batch_size=1,
    hidden_dim=None,
    noise=0.01,
    log_path="results/experiment_log.csv",
    metadata=None,
    alpha_nn=None,
    beta_nn=None,
    gamma_nn=None,
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    results = {}

    if model_type == 'regression':
        from src.models.regression_model import RegressionModel
        from src.sgd.sgd import SGD

        model = RegressionModel(X_train, y_train)
        optimizer = SGD(model, num_iterations=num_iterations, batch_size=batch_size, noise=noise, stepsize_type=stepsize_strategy)
        params = optimizer.params  
        w, obj_hist, grad_hist, dist_hist = optimizer.optimize()
        preds = X_test @ w[1:] + w[0]
        test_loss = np.mean((preds - y_test.flatten())**2)

    elif model_type == 'nn':
        from src.models.twolayer_nn import TwoLayerNNModel
        from src.sgd.nonconvex import NonConvexSGD, FixedStepsize, DiminishingStepsize

        assert hidden_dim is not None, "Specify hidden_dim for nns."
        model = TwoLayerNNModel(input_dim=X_train.shape[1], hidden_dim=hidden_dim)

        if stepsize_strategy == 'fixed':
            alpha = alpha_nn if alpha_nn is not None else 0.01
            strategy = FixedStepsize(alpha)
        elif stepsize_strategy == 'diminishing':
            beta, gamma = beta_nn if beta_nn is not None else 1, gamma_nn if gamma_nn is not None else 100
            strategy = DiminishingStepsize(beta, gamma)
        else:
            raise NotImplementedError("Halving not supported for NN.")

        optimizer = NonConvexSGD(model, X_train, y_train, num_iterations=num_iterations, batch_size=batch_size, stepsize_type=strategy)
        w, obj_hist, grad_hist = optimizer.optimize()
        preds = np.array([model.forward(x.flatten(), w) for x in X_test])
        test_loss = np.mean((preds - y_test.flatten())**2)

        params = {}  

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Main experiment result logging
    results.update({
        "model_type": model_type,
        "stepsize_strategy": stepsize_strategy,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim if hidden_dim else -1,
        "test_loss": test_loss,
        "train_loss": obj_hist[-1],
        "grad_norm": grad_hist[-1],
        "dist_to_opt": dist_hist[-1],
    })

    if metadata:
        results.update(metadata)

    df = pd.DataFrame([results])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)

    # -----------------------------
    # Parameter logging (if available)
    # -----------------------------
    if params:
        basename = os.path.basename(log_path).replace(".csv", "")
        param_log_path = os.path.join(os.path.dirname(log_path), f"{basename}_params.csv")
        os.makedirs(os.path.dirname(param_log_path), exist_ok=True)
        param_record = params.copy()
        if metadata:
            param_record.update(metadata)
        pd.DataFrame([param_record]).to_csv(param_log_path, mode='a', index=False, header=not os.path.exists(param_log_path))

    return results


def main(
    X_train,
    y_train,
    X_test,
    y_test,
    model_type="regression",
    stepsize_strategy="fixed",
    num_iterations=1000,
    batch_size=1,
    hidden_dim=None,
    noise=0.01,
    log_path="results/experiment_log.csv",
    metadata=None,
    alpha_nn=None,
    beta_nn=None,
    gamma_nn=None,
):
    return run_experiment_and_log(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        stepsize_strategy=stepsize_strategy,
        num_iterations=num_iterations,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        noise=noise,
        log_path=log_path,
        metadata=metadata,
        alpha_nn=alpha_nn,
        beta_nn=beta_nn,
        gamma_nn=gamma_nn
    )
