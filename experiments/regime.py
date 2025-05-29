import sys
sys.path.append("..")

import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from src.data.generate_synthetic_data import generate_training_data_fixed, transform_to_polynomial

# ---- Build model ----
def build_model(input_dim, width, depth):
    layers = [nn.Linear(input_dim, width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)

# ---- Training function ----
def run_training(X_train, y_train, X_test, y_test, width, depth, stepsize, epochs=1000):
    model = build_model(X_train.shape[1], width, depth)
    optimizer = optim.SGD(model.parameters(), lr=stepsize)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.step()

    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()

    return loss.item(), grad_norm, test_loss

# ---- Global settings ----
poly_degree = 3
input_dim = 2
noise_std = 0.001
num_trials = 5

depths = [1, 5]
widths = [1, 5, 10, 20, 50]
stepsizes = [1e-3, 1e-2, 0.002336524216440747, 0.1]
sample_sizes = [5, 8, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 7000]

# ---- Target function coefficients ----
X_dummy = np.random.normal(0, 1, size=(10, input_dim))
X_dummy_poly = transform_to_polynomial(X_dummy, degree=poly_degree, normalize=True)
true_A = 0.1 * np.arange(1, X_dummy_poly.shape[1] + 1)
true_b = 1.0

results = []

# ---- Sweep over regimes and seeds ----
for d in depths:
    for w in widths:
        for eta in stepsizes:
            for n in sample_sizes:
                train_losses = []
                grad_norms = []
                test_losses = []
                for seed in range(num_trials):
                    np.random.seed(0)
                    torch.manual_seed(0)
                    X_raw, _, _ = generate_training_data_fixed(m=n, n=input_dim, noise=noise_std)
                    X_poly = transform_to_polynomial(X_raw, degree=poly_degree, normalize=True)
                    y = X_poly @ true_A + true_b + np.random.normal(0, noise_std, size=X_poly.shape[0])

                    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3)
                    X_train = torch.tensor(X_train, dtype=torch.float32)
                    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                    X_test = torch.tensor(X_test, dtype=torch.float32)
                    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

                    train_loss, grad_norm, test_loss = run_training(
                        X_train, y_train, X_test, y_test, width=w, depth=d, stepsize=eta
                    )

                    train_losses.append(train_loss)
                    grad_norms.append(grad_norm)
                    test_losses.append(test_loss)

                results.append({
                    'depth': d,
                    'width': w,
                    'stepsize': eta,
                    'n_samples': n,
                    'train_loss_mean': np.mean(train_losses),
                    'grad_norm_mean': np.mean(grad_norms),
                    'test_loss_mean': np.mean(test_losses),
                })

# ---- Save results ----
df = pd.DataFrame(results)
df.to_csv("regime_results.csv", index=False)
print("Saved results to regime_results.csv")

# ---- Plotting ----
def plot_test_loss_vs_variable(df, variable, fixed_conditions, save_name):
    subset = df.copy()
    for k, v in fixed_conditions.items():
        if np.issubdtype(subset[k].dtype, np.floating):
            subset = subset[np.isclose(subset[k], v)]
        else:
            subset = subset[subset[k] == v]
    subset = subset.sort_values(by=variable)
    plt.figure()
    plt.plot(subset[variable], subset['test_loss_mean'], marker='o')
    plt.xlabel(variable)
    plt.ylabel('Test Loss')
    title = f"Test Loss vs {variable} | " + ", ".join(f"{k}={v}" for k, v in fixed_conditions.items())
    plt.title(title)
    if variable == 'stepsize' or variable == 'n_samples':
        plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    filepath = f"plots/{save_name}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

# ---- Run plots ----
fixed_config = {'depth': 5, 'width': 10, 'n_samples': 1000, 'stepsize': 1e-2}

plot_test_loss_vs_variable(df, 'stepsize', {k: v for k, v in fixed_config.items() if k != 'stepsize'}, "test_loss_vs_stepsize")
plot_test_loss_vs_variable(df, 'depth', {k: v for k, v in fixed_config.items() if k != 'depth'}, "test_loss_vs_depth")
plot_test_loss_vs_variable(df, 'width', {k: v for k, v in fixed_config.items() if k != 'width'}, "test_loss_vs_width")
plot_test_loss_vs_variable(df, 'n_samples', {k: v for k, v in fixed_config.items() if k != 'n_samples'}, "test_loss_vs_n_samples")
print("Saved plots to /plots/")
