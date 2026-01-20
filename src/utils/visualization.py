import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

def plot_grid_search_best_16(all_results, relative_path="results/grid_search/plots"):
    """
    Plotta i 16 migliori risultati della Grid Search in una griglia 4x4.
    Solo MSE (Loss) su Training e Validation.
    
    Args:
        all_results (list): Lista di dizionari. Ogni dizionario deve avere:
                            {
                                'params': dict (es. {'lr': 0.01, ...}),
                                'tr_mse': list (history),
                                'vl_mse': list (history)
                            }
        relative_path (str): Dove salvare il file.
    """
    dir_path = _ensure_dir(relative_path)
    
    # 1. Ordina i risultati in base al MINIMO MSE di validazione raggiunto
    #    (Assumiamo che 'vl_mse' sia una lista, prendiamo il min)
    sorted_results = sorted(all_results, key=lambda x: min(x['vl_mse']))
    
    # Prendiamo i primi 16 (o meno se ce ne sono meno)
    top_n = min(16, len(sorted_results))
    top_results = sorted_results[:top_n]
    
    # 2. Setup della figura 4x4
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 18)) # Molto grande per leggibilità
    axes = axes.flatten() # Appiattiamo per iterare facilmente

    # Titolo globale
    plt.suptitle(
        f"Top {top_n} Configurations (Sorted by Validation MSE) - {datetime.now().strftime('%d/%m %H:%M')}", 
        fontsize=20, 
        fontweight='bold',
        y=0.98
    )

    for i in range(rows * cols):
        ax = axes[i]
        
        if i < top_n:
            res = top_results[i]
            tr_hist = res['tr_mse']
            vl_hist = res['vl_mse']
            params = res['params']
            
            epochs = np.arange(1, len(tr_hist) + 1)
            
            # Calcolo best epoch per il marker
            best_vl_val = min(vl_hist)
            best_epoch = np.argmin(vl_hist) + 1
            
            # --- PLOT TRAINING (Rosso) ---
            ax.plot(epochs, tr_hist, color='red', label='Train', linewidth=1.5, alpha=0.8)
            
            # --- PLOT VALIDATION (Blu Tratteggiato) ---
            ax.plot(epochs, vl_hist, color='blue', linestyle='--', label='Val', linewidth=1.5)
            
            # --- MARKER BEST (Stella Verde) ---
            ax.scatter(best_epoch, best_vl_val, color='green', marker='*', s=80, zorder=5)

            # --- FORMATTAZIONE ASSI ---
            ax.set_yscale('log') # Fondamentale per MSE
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Titolo del singolo subplot (Parametri formattati)
            # Converte dizionario parametri in stringa breve: "lr=0.1, mom=0.5..."
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            
            # Se la stringa è troppo lunga, la tronchiamo o andiamo a capo
            if len(param_str) > 40:
                param_str = param_str[:40] + "..."
                
            ax.set_title(f"#{i+1}: {param_str}\nBest VL: {best_vl_val:.5f}", fontsize=10, fontweight='bold')
            
            # Label solo sui bordi esterni per pulizia
            if i >= 12: # Ultima riga
                ax.set_xlabel("Epochs")
            if i % 4 == 0: # Prima colonna
                ax.set_ylabel("MSE (log)")
                
        else:
            # Nascondi assi vuoti se ci sono meno di 16 risultati
            ax.axis('off')

    # Legenda unica in basso (opzionale, o per ogni plot se preferisci)
    # Qui ne mettiamo una finta sul primo grafico per riferimento
    axes[0].legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Lascia spazio per il titolo su
    
    # Salvataggio
    fname = _generate_filename("GRID_Top16_MSE")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=200) # dpi leggermente più basso per file non enormi
    
    print(f"✅ Grid Search Top 16 salvata in: {full_path}")

def plot_grid_analysis(all_results, top_k_individual=5, relative_path="results/grid_search"):
    """
    Analizza i risultati della Grid Search e produce:
    1. Un 'Summary Plot' 4x4 con i migliori 16 modelli (solo MSE).
    2. Una collezione di plot individuali per i migliori 'top_k_individual' modelli.
    
    Args:
        all_results (list): Lista di dizionari {params, tr_mse, vl_mse}.
        top_k_individual (int): Quanti grafici singoli salvare (es. i migliori 5).
        relative_path (str): Cartella base di salvataggio.
    """
    
    # 1. Setup Cartelle
    base_dir = _ensure_dir(relative_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sottocartella per i plot singoli per non intasare la root
    single_plots_dir = os.path.join(base_dir, f"details_{timestamp}")
    if top_k_individual > 0:
        os.makedirs(single_plots_dir, exist_ok=True)

    # 2. Ordinamento: dal MSE di validazione più basso (migliore) al più alto
    #    Assumiamo che tr_mse e vl_mse siano liste storiche
    sorted_results = sorted(all_results, key=lambda x: min(x['vl_mse']))
    
    # ==========================================
    # PARTE A: SUMMARY GRID (4x4)
    # ==========================================
    top_16 = sorted_results[:16]
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    plt.suptitle(f"Top 16 Configurations (Sorted by Val MSE) - {timestamp}", fontsize=18, fontweight='bold')
    
    for i in range(rows * cols):
        ax = axes[i]
        
        if i < len(top_16):
            res = top_16[i]
            _draw_single_mse_plot(ax, res, title_prefix=f"Rank #{i+1}")
            
            # Label assi solo sui bordi esterni per pulizia
            if i >= 12: ax.set_xlabel("Epoch")
            if i % 4 == 0: ax.set_ylabel("MSE (log)")
        else:
            ax.axis('off') # Nascondi riquadri vuoti

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    summary_fname = f"GRID_Summary_Top16_{timestamp}.png"
    summary_path = os.path.join(base_dir, summary_fname)
    plt.savefig(summary_path, dpi=200)
    print(f"✅ [Summary] Griglia 4x4 salvata in: {summary_path}")
    plt.close()

    # ==========================================
    # PARTE B: COLLEZIONE INDIVIDUALE (Top K)
    # ==========================================
    if top_k_individual > 0:
        print(f"   ...Generazione di {top_k_individual} grafici individuali...")
        
        for i in range(min(top_k_individual, len(sorted_results))):
            res = sorted_results[i]
            
            # Creiamo una figura nuova per ogni plot
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            
            # Disegna
            params_str = _format_params(res['params'])
            _draw_single_mse_plot(ax_single, res, title_prefix=f"Rank #{i+1}", font_scale=1.2)
            
            # Aggiungi Info Box dettagliata solo nel plot singolo
            final_tr = res['tr_mse'][-1]
            best_vl = min(res['vl_mse'])
            info_text = (f"Params: {params_str}\n"
                         f"Final TR MSE: {final_tr:.5e}\n"
                         f"Best VL MSE: {best_vl:.5e}")
            
            ax_single.text(0.5, -0.2, info_text, ha='center', va='top', transform=ax_single.transAxes,
                           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
            
            ax_single.set_xlabel("Epoch")
            ax_single.set_ylabel("MSE (log)")
            ax_single.set_title(f"Configuration Rank #{i+1}", fontweight='bold')
            
            plt.tight_layout()
            
            # Nome file pulito
            safe_params = params_str.replace(", ", "_").replace("=", "").replace(".", "")[:50]
            fname_single = f"Rank_{i+1:02d}_{safe_params}.png"
            plt.savefig(os.path.join(single_plots_dir, fname_single), dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"✅ [Collection] {top_k_individual} grafici dettagliati salvati in: {single_plots_dir}")


def _draw_single_mse_plot(ax, result, title_prefix="", font_scale=1.0):
    """Funzione helper per disegnare lo stile 'Monk' su un asse dato."""
    tr_hist = result['tr_mse']
    vl_hist = result['vl_mse']
    params = result['params']
    epochs = np.arange(1, len(tr_hist) + 1)
    
    best_vl = min(vl_hist)
    best_epoch = np.argmin(vl_hist) + 1
    
    # 1. Plot Training (Rosso)
    ax.plot(epochs, tr_hist, color='red', label='Train MSE', linewidth=1.5 * font_scale, alpha=0.9)
    
    # 2. Plot Validation (Blu Tratteggiato)
    ax.plot(epochs, vl_hist, color='blue', linestyle='--', label='Val MSE', linewidth=1.5 * font_scale, alpha=0.9)
    
    # 3. Stella Verde sul punto migliore
    ax.scatter(best_epoch, best_vl, color='green', marker='*', s=100 * font_scale, zorder=5, edgecolors='black')
    
    # Stile
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Titolo breve
    param_str = _format_params(params)
    if len(param_str) > 40 * font_scale: 
        param_str = param_str[:int(40*font_scale)] + "..."
    
    ax.set_title(f"{title_prefix}\n{param_str}\nBest VL: {best_vl:.5f}", fontsize=9 * font_scale, fontweight='bold')
    
    # Legenda (piccola per non coprire)
    ax.legend(fontsize=8 * font_scale, loc='upper right')

def _format_params(params):
    """Converte dizionario parametri in stringa leggibile: 'lr=0.01, mom=0.9'"""
    return ", ".join([f"{k}={v}" for k, v in params.items()])

def plot_errors_with_validation_error(
    trainer,
    training_time: float,
    relative_path="results/cup/plots"
):
    dir_path = _ensure_dir(relative_path)

    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    # Valori finali
    final_tr_mee = trainer.tr_mee_history[-1]
    final_vl_mee = trainer.vl_mee_history[-1]
    best_vl_mee = min(trainer.vl_mee_history)
    best_epoch = np.argmin(trainer.vl_mee_history) + 1
    
    # Setup Figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ==========================
    # GRAFICO 1: MEE (Metrica Principale)
    # ==========================
    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        color='red',           # STILE MONK: Training rosso
        linewidth=2,
        label=f"MEE_tr (final={final_tr_mee:.4f})",
        alpha=0.9
    )

    ax1.plot(
        epochs,
        trainer.vl_mee_history,
        color='blue',          # STILE MONK: Validation blu
        linestyle="--",        # STILE MONK: Tratteggiato
        label=f"MEE_vl (best={best_vl_mee:.4f})",
        linewidth=2
    )

    # Marker per il punto migliore (Stella Verde come Monk)
    ax1.scatter(
        best_epoch,
        best_vl_mee,
        color="green",         # STILE MONK: Stella verde
        marker='*',
        s=150,
        edgecolors='black',
        zorder=5,
        label=f"Best vl @ ep {best_epoch}"
    )

    ax1.set_title("Mean Euclidean Error (MEE)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MEE", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=11)

    # ==========================
    # GRAFICO 2: MSE (Loss)
    # ==========================
    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        color='red',           # STILE MONK
        label="MSE_tr",
        alpha=0.9
    )

    ax2.plot(
        epochs,
        trainer.vl_mse_history,
        color='blue',          # STILE MONK
        linestyle="--",
        label="MSE_vl",
        alpha=0.9
    )

    ax2.set_title("Mean Squared Error (Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MSE (log scale)", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=11)

    # ==========================
    # INFO BOX (Stile Monk)
    # ==========================
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
        fontsize=12,
        # STILE MONK: Sfondo lightblue invece di wheat
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3) 
    )

    plt.suptitle(
        f"CUP Training Analysis ({datetime.now().strftime('%d/%m/%Y')})",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    fname = _generate_filename("CUP_tr_vs_vl")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300)

    print(f"✅ Grafico CUP salvato in: {full_path}")


def plot_errors(trainer, training_time: float, relative_path="results/cup/plots"):
    """
    Plot solo training (no validation) - Stile adattato al Monk (tutto Rosso)
    """
    dir_path = _ensure_dir(relative_path)
    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # MEE
    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        color='red',
        linewidth=2,
        label=f"MEE_tr (final={trainer.tr_mee_history[-1]:.4f})"
    )
    ax1.set_title("Training MEE", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MEE")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # MSE
    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        color='red',
        linewidth=2,
        label=f"MSE_tr (final={trainer.tr_mse_history[-1]:.4f})"
    )
    ax2.set_title("Training MSE (Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    # Info Box semplificato
    info_text = f"Final MEE: {trainer.tr_mee_history[-1]:.5f} | Time: {training_time:.2f}s"
    fig.text(
        0.5, 0.02,
        info_text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3)
    )

    plt.suptitle(
        f"CUP Training Profile (No Validation)", 
        fontsize=16, 
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    fname = _generate_filename("CUP_train_only")
    plt.savefig(os.path.join(dir_path, fname))

    print(f"✅ Grafico CUP (TrainOnly) salvato in: {os.path.join(dir_path, fname)}")


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