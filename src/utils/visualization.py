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
    Versione avanzata con Training vs Validation, statistiche finali e salvataggio automatico.
    """
    dir_path = _ensure_dir(save_path)
    epochs = np.arange(1, len(trainer.mee_error_history) + 1)
    
    # Recupero i valori finali per i dettagli
    final_tr_mee = trainer.mee_error_history[-1]
    final_vl_mee = trainer.mee_vl_error_history[-1]
    best_vl_mee = min(trainer.mee_vl_error_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Grafico 1: MEE (Focus sulla metrica CUP) ---
    ax1.plot(epochs, trainer.mee_error_history, label=f"Train (Final: {final_tr_mee:.4f})", color='#1f77b4', alpha=0.8)
    ax1.plot(epochs, trainer.mee_vl_error_history, label=f"Val (Best: {best_vl_mee:.4f})", color='#ff7f0e', linewidth=2)
    ax1.set_title("Mean Euclidean Error (MEE)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MEE")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Aggiungiamo un'evidenziazione sul punto minimo della validation
    best_epoch = np.argmin(trainer.mee_vl_error_history) + 1
    ax1.scatter(best_epoch, best_vl_mee, color='red', zorder=5, label='Best Val')

    # --- Grafico 2: MSE (Focus sulla convergenza) ---
    ax2.plot(epochs, trainer.mse_error_history, label="MSE Train", color='#2ca02c', alpha=0.7)
    ax2.plot(epochs, trainer.mse_vl_error_history, label="MSE Validation", color='#d62728', alpha=0.7, linestyle='--')
    ax2.set_title("Mean Squared Error (Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_yscale('log') # Scala logaritmica spesso migliore per la loss
    ax2.set_ylabel("MSE (Log Scale)")
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend()

    # --- Info Box & Titolo ---
    info_text = (f"Final Validation MEE: {final_vl_mee:.5f}\n"
                 f"Best Validation MEE: {best_vl_mee:.5f} (Ep: {best_epoch})\n"
                 f"Training Time: {training_time:.2f}s")
    
    # Inserisce il testo informativo nel grafico
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.3))

    plt.suptitle(f"Neural Network Training Analysis - {datetime.now().strftime('%d/%m/%Y')}", 
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Salvataggio
    fname = _generate_filename("CUP_validation")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Grafico salvato in: {full_path}")
    
    plt.show()

def plot_errors(trainer, training_time: float, save_path="../results/plots"):
    """Versione per training singolo (senza validation)."""
    dir_path = _ensure_dir(save_path)
    epochs = np.arange(1, len(trainer.mee_error_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, trainer.mee_error_history, color='#1f77b4', label=f"Final MEE: {trainer.mee_error_history[-1]:.4f}")
    ax1.set_title("Training MEE")
    ax1.set_xlabel("Epochs")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, trainer.mse_error_history, color='#d62728', label=f"Final MSE: {trainer.mse_error_history[-1]:.4f}")
    ax2.set_title("Training MSE (Loss)")
    ax2.set_xlabel("Epochs")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Training Profile - Execution Time: {training_time:.2f}s")
    plt.tight_layout()
    
    fname = _generate_filename("CUP_train_only")
    plt.savefig(os.path.join(dir_path, fname))
    plt.show()