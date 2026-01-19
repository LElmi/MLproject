import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)



def plot_errors_with_validation_error(
    trainer,
    training_time: float,
    relative_path="results/plots"
):
    dir_path = _ensure_dir(relative_path)

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


def plot_errors(trainer, training_time: float, relative_path="results/plots"):
    """
    Plot solo training (no validation)
    """
    dir_path = _ensure_dir(relative_path)
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

def plot_accuracy(trainer, training_time: float, relative_path="results/plots"):
    
    dir_path = _ensure_dir(relative_path)
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

def _ensure_dir(relative_path="results/plots"):
    """
    Crea una directory RELATIVA ALLA CARTELLA DEL PROGETTO
    """

    PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

    print("PROJECT_ROOT", PROJECT_ROOT)

    full_path = os.path.join(PROJECT_ROOT, relative_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path



def _generate_filename(prefix="plot"):
    """Genera un nome file unico basato sul timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"



def plot_monk(
    tr_mse_history: list,
    tr_accuracy_history: list,
    #ts_mse_history: list,
    ts_accuracy_history: list,
    training_time: float,
    relative_path="results/monk/plots"
):
    """
    Plot per MONK con MSE e Accuracy su Training e Test Set.
    
    Args:
        tr_mse_history: Lista MSE sul training set
        tr_accuracy_history: Lista accuracy sul training set
        ts_mse_history: Lista MSE sul test set
        ts_accuracy_history: Lista accuracy sul test set
        training_time: Tempo di training in secondi
        relative_path: Percorso relativo per salvare il plot
    """
    dir_path = _ensure_dir(relative_path)
    
    epochs = np.arange(1, len(tr_mse_history) + 1)
    
    # Valori finali
    final_tr_mse = tr_mse_history[-1]
    #final_ts_mse = ts_mse_history[-1]
    final_tr_acc = tr_accuracy_history[-1]
    final_ts_acc = ts_accuracy_history[-1]
    
    # Valori migliori
    best_ts_acc = max(ts_accuracy_history)
    best_epoch = np.argmax(ts_accuracy_history) + 1
    
    # Crea figura con 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== GRAFICO 1: MSE ==========
    ax1.plot(
        epochs,
        tr_mse_history,
        color='red',
        linewidth=2,
        label=f'MSE Training (final={final_tr_mse:.4f})'
    )
    
    """
    ax1.plot(
        epochs,
        ts_mse_history,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'MSE Test (final={final_ts_mse:.4f})'
    )"""
    
    ax1.set_title("Mean Squared Error (Loss)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=11)
    
    # ========== GRAFICO 2: ACCURACY ==========
    ax2.plot(
        epochs,
        tr_accuracy_history,
        color='red',
        linewidth=2,
        label=f'Accuracy Training (final={final_tr_acc*100:.2f}%)'
    )
    
    ax2.plot(
        epochs,
        ts_accuracy_history,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Accuracy Test (best={best_ts_acc*100:.2f}%)'
    )
    
    # Evidenzia il punto migliore sul test set
    ax2.scatter(
        best_epoch,
        best_ts_acc,
        color='green',
        s=150,
        zorder=5,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label=f'Best @ epoch {best_epoch}'
    )
    
    # Linea al 100%
    ax2.axhline(
        y=1.0,
        color='gray',
        linestyle=':',
        alpha=0.5,
        linewidth=1.5,
        label='100% Target'
    )
    
    ax2.set_title("Classification Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim([0, 1.05])  # Da 0% a 105%
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=11)
    
    # ========== INFO BOX ==========
    info_text = (
        #f"Training MSE: {final_tr_mse:.5f} | Test MSE: {final_ts_mse:.5f}\n"
        f"Training MSE: {final_tr_mse:.5f}"
        f"Training Acc: {final_tr_acc*100:.2f}% | Test Acc: {final_ts_acc*100:.2f}%\n"
        f"Best Test Acc: {best_ts_acc*100:.2f}% @ epoch {best_epoch}\n"
        f"Training time: {training_time:.2f}s"
    )
    
    fig.text(
        0.5, 0.02,
        info_text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3)
    )
    
    # ========== TITOLO ==========
    plt.suptitle(
        f"MONK Classification Results - {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        fontsize=16,
        fontweight="bold"
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    # ========== SALVATAGGIO ==========
    fname = _generate_filename("MONK_complete")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Grafico MONK salvato in: {full_path}")
    plt.close()