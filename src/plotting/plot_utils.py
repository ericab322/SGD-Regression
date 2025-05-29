import matplotlib.pyplot as plt
import numpy as np
import os

def plot_phase_transition(results, d, l, s, stepsize_methods, save_dir=None):
    """
    Plots test loss vs. number of samples for given (d, l) using multiple stepsize strategies.

    Args:
        results: results dictionary.
        d: Input dimension.
        l: Polynomial degree.
        s: Parameter for phase thresholds.
        stepsize_methods:["fixed", "halving", "diminishing"].
        save_dir: Optional directory to save the figure. If None, plot is not saved.
    """
    ns = results[(d, l)][stepsize_methods[0]]["ns"]

    plt.figure(figsize=(10, 6))
    for method in stepsize_methods:
        test_losses = results[(d, l)][method]["test"]
        plt.plot(ns, test_losses, marker='o', label=f"{method.capitalize()} Stepsize")
        
    lower_n = d ** (l + s)
    upper_n = (d + 1) ** (l + 1 - s)

    plt.axvline(lower_n, color='red', linestyle='--', label=rf"$n = d^{{l+s}} = {int(lower_n)}$")
    plt.axvline(upper_n, color='green', linestyle='--', label=rf"$n = (d+1)^{{l+1-s}} = {int(upper_n)}$")

    plt.axvspan(ns[0], lower_n, color='red', alpha=0.1)
    plt.axvspan(lower_n, upper_n, color='yellow', alpha=0.1)
    plt.axvspan(upper_n, ns[-1], color='green', alpha=0.1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Test Loss (MSE)")
    plt.title(f"Phase Transition — d={d}, degree={l}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save plot if path is given
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"phase_transition_d{d}_l{l}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
        print(f"Saved plot to {os.path.join(save_dir, filename)}")
    else:
        plt.show()
        plt.close()
