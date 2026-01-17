import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

def _ensure_dir(path="../results/plots"):
    """Assicura che la cartella di destinazione esista."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _generate_filename(prefix="plot"):
    """Genera un nome file unico basato sul timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"

def plot_errors_with_validation_error(trainer, training_time: float, save_path="../results/plots"):
    """
    Plot Training vs Validation con naming coerente (tr / vl)
    """
    dir_path = _ensure_dir(save_path)

    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    # Valori finali
    final_tr_mee = trainer.tr_mee_history[-1]
    final_vl_mee = trainer.vl_mee_history[-1]
    best_vl_mee = min(trainer.vl_mee_history)
    best_epoch = np.argmin(trainer.vl_mee_history) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---------- GRAFICO 1: MEE ----------
    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        label=f"MEE_tr (final={final_tr_mee:.4f})",
        alpha=0.8
    )

    ax1.plot(
        epochs,
        trainer.vl_mee_history,
        label=f"MEE_vl (best={best_vl_mee:.4f})",
        linewidth=2
    )

    ax1.scatter(
        best_epoch,
        best_vl_mee,
        color="red",
        zorder=5,
        label=f"Best vl @ ep {best_epoch}"
    )

    ax1.set_title("Mean Euclidean Error (MEE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MEE")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # ---------- GRAFICO 2: MSE ----------
    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        label="MSE_tr",
        alpha=0.8
    )

    ax2.plot(
        epochs,
        trainer.vl_mse_history,
        label="MSE_vl",
        linestyle="--",
        alpha=0.8
    )

    ax2.set_title("Mean Squared Error (Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    # ---------- INFO BOX ----------
    info_text = (
        f"Final MEE_tr: {final_tr_mee:.5f}\n"
        f"Final MEE_vl: {final_vl_mee:.5f}\n"
        f"Best MEE_vl: {best_vl_mee:.5f} (ep {best_epoch})\n"
        f"Training time: {training_time:.2f}s"
    )

    fig.text(
        0.5, 0.02,
        info_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    )

    plt.suptitle(
        f"Training Analysis ({datetime.now().strftime('%d/%m/%Y')})",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    fname = _generate_filename("CUP_tr_vs_vl")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300)

    print(f"✅ Grafico salvato in: {full_path}")


def plot_errors(trainer, training_time: float, save_path="../results/plots"):
    """
    Plot solo training (no validation)
    """
    dir_path = _ensure_dir(save_path)
    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        label=f"MEE_tr (final={trainer.tr_mee_history[-1]:.4f})"
    )
    ax1.set_title("Training MEE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MEE")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        label=f"MSE_tr (final={trainer.tr_mse_history[-1]:.4f})"
    )
    ax2.set_title("Training MSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Training Profile – time: {training_time:.2f}s")
    plt.tight_layout()

    fname = _generate_filename("CUP_train_only")
    plt.savefig(os.path.join(dir_path, fname))

    print(f"✅ Grafico salvato in: {os.path.join(dir_path, fname)}")

def plot_accuracy(trainer, training_time: float, save_path="../results/plots"):
        dir_path = _ensure_dir(save_path)
        epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(
            epochs,
            trainer.tr_mee_history,
            label=f"MEE_tr (final={trainer.accuracy_history[-1]:.4f})"
        )
        ax1.set_title("Training accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("accuracy")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(
            epochs,
            trainer.tr_mse_history,
            label=f"MSE_tr (final={trainer.accuracy_history[-1]:.4f})"
        )
        ax2.set_title("Training MSE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(f"Training Profile – time: {training_time:.2f}s")
        plt.tight_layout()

        fname = _generate_filename("Monk_accuracy_training")
        plt.savefig(os.path.join(dir_path, fname))

        print(f"✅ Grafico salvato in: {os.path.join(dir_path, fname)}")
