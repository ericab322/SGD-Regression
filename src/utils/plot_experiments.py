import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Style setup
sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette("Set2") 

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True
})

# Optional smoothing for convergence curves
def smooth(y, box_pts=5):
    if len(y) <= box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

# Model complexity plot
def plot_model_complexity(df, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_var = "degree" if df["degree"].notna().any() else "hidden_dim"
    x_vals = sorted(df[x_var].dropna().unique())

    for idx, method in enumerate(df["stepsize_strategy"].unique()):
        subset = df[df["stepsize_strategy"] == method]
        means = subset.groupby(x_var)["test_loss"].mean().reindex(x_vals)
        ax.plot(
            x_vals, means.values,
            marker="o", markersize=8, linewidth=2.5,
            color=palette[idx % len(palette)],
            label=method.capitalize()
        )

    ax.set_xlabel("Model Degree" if x_var == "degree" else "Hidden Layer Size", labelpad=10)
    ax.set_ylabel("Test MSE", labelpad=10)
    ax.set_title("Model Complexity vs Test MSE", pad=15)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_mse_vs_complexity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "test_mse_vs_complexity.pdf"), bbox_inches="tight")
    plt.close()

# Sample complexity plot
def plot_sample_complexity(df, save_dir, s=0.0):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = sorted(df["n"].unique())

    for idx, method in enumerate(df["stepsize_strategy"].unique()):
        subset = df[df["stepsize_strategy"] == method]
        means = subset.groupby("n")["test_loss"].mean().reindex(x_vals)
        ax.plot(
            x_vals, means.values,
            marker='o', markersize=8, linewidth=2.5,
            color=palette[idx % len(palette)],
            label=method.capitalize()
        )

    ax.set_xlabel("Sample Size", labelpad=10)
    ax.set_ylabel("Test MSE", labelpad=10)
    ax.set_title("Sample Complexity vs Test MSE", pad=15)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_mse_vs_sample_size.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "test_mse_vs_sample_size.pdf"), bbox_inches="tight")
    plt.close()


def plot_convergence(df, save_dir):
    for idx, (_, row) in enumerate(df.iterrows()):
        obj_hist = smooth(json.loads(row["obj_history"]))
        grad_hist = smooth(json.loads(row["grad_norm_history"]))

        title_str = (
            f"{row['experiment_type'].replace('_',' ').title()} | "
            f"{row['stepsize_strategy'].capitalize()} | n={row['n']}"
        )
        if pd.notna(row.get("degree", None)):
            title_str += f" | deg={int(row['degree'])}"
        if pd.notna(row.get("hidden_dim", None)):
            title_str += f" | hid={int(row['hidden_dim'])}"

        color = palette[idx % len(palette)]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(obj_hist)), obj_hist, linewidth=2.5, color=color)
        ax.set_xlabel("Iteration", labelpad=10)
        ax.set_ylabel("Train Loss", labelpad=10)
        ax.set_title(f"Convergence: Train Loss\n{title_str}", pad=15)
        plt.tight_layout()
        fname = f"convergence_loss_{row['stepsize_strategy']}_n{row['n']}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(grad_hist)), grad_hist, linewidth=2.5, color=color)
        ax.set_xlabel("Iteration", labelpad=10)
        ax.set_ylabel("Gradient Norm", labelpad=10)
        ax.set_title(f"Convergence: Gradient Norm\n{title_str}", pad=15)
        plt.tight_layout()
        fname = f"convergence_grad_{row['stepsize_strategy']}_n{row['n']}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

# Main
def main(csv_path):
    df = pd.read_csv(csv_path)
    save_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(save_dir, exist_ok=True)

    exp_type = df["experiment_type"].iloc[0]

    if "model_complexity" in exp_type:
        plot_model_complexity(df, save_dir)
    elif "sample_complexity" in exp_type:
        plot_sample_complexity(df, save_dir)

    plot_convergence(df, save_dir)
    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_experiments.py path/to/experiment_log.csv")
        sys.exit(1)
    main(sys.argv[1])
