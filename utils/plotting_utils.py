import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history, filename="plots/loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Training Loss", color="navy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Adam + Dropout + L2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")

def plot_predictions(true_vals, preds, filename="plots/predicted_vs_true.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.scatter(true_vals[:, i], preds[:, i], c="blue", alpha=0.7, label="Predicted Points")
        ax.plot(
            [true_vals[:, i].min(), true_vals[:, i].max()],
            [true_vals[:, i].min(), true_vals[:, i].max()],
            "r--", label="Perfect Fit (y = x)"
        )
        ax.set_title(f"Target {i + 1} - Predicted vs True")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Model Predictions vs True Targets", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Saved: {filename}")
