"""
Generates and saves training loss plots and raw data
for post training analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(losses, run_path):
    """
    Generate and save a training loss plot.

    @param losses List of per epoch loss values.
    @param run_path Path to the run folder where plots will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(losses, alpha=0.4, linewidth=0.8, label="Loss")

    # Moving average for trend clarity
    window = min(50, len(losses) // 10)
    if window > 0:
        moving_avg = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(losses)),
            moving_avg,
            color="red",
            linewidth=2,
            label=f"{window} epoch moving avg",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_dir = os.path.join("plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Save raw data alongside the plot
    data_path = os.path.join(run_path, "training_data.npz")
    np.savez(data_path, losses=losses)

    print(f"Plots saved to {plot_path}")
