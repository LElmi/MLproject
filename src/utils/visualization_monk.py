import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

def plot_grid_search_analysis_monk2(all_results, relative_path="results/monk/grid_search", top_k=6):
    
    # Creazione cartelle
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_dir = os.path.join(relative_path, f"details_{timestamp}")
    if not os.path.exists(details_dir):
        os.makedirs(details_dir)
    
    # Ordiniamo i risultati (migliore accuracy prima)
    sorted_res = sorted(all_results, key=lambda x: max(x['vl_accuracy']), reverse=True)
    top_models = sorted_res[:top_k]
    
    # ---------------------------------------------------------
    # 1. GRIGLIA RIASSUNTIVA (Summary)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    plt.suptitle("Top 6 Modelli MONK", fontsize=16, fontweight='bold')
    
    for i in range(6):
        ax = axes[i]
        
        if i < len(top_models):
            res = top_models[i]
            mse = res['tr_mse']
            best_acc = max(res['vl_accuracy'])
            epochs = range(1, len(mse) + 1)
            
            # Plot semplice dell'MSE
            ax.plot(epochs, mse, color='red', label='Train MSE')
            
            # Titolo standard
            ax.set_title(f"Rank #{i+1} - Val Acc: {best_acc*100:.1f}%", fontsize=10, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Testo parametri piccolo (formattato stretto per la griglia)
            params_str = format_params(res['params'], layout='grid')
            ax.text(0.5, -0.15, params_str, transform=ax.transAxes, 
                    ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.4)
    
    summary_name = f"Summary_Top6_{timestamp}.png"
    plt.savefig(os.path.join(relative_path, summary_name))
    plt.close()
    print(f"✅ Grafico sommario salvato in: {relative_path}")

    # ---------------------------------------------------------
    # 2. DETTAGLI SINGOLI (Orizzontali)
    # ---------------------------------------------------------
    print(f"   ...Generazione dei {len(top_models)} grafici di dettaglio...")
    
    for i, res in enumerate(top_models):
        plot_single_detail(res, i+1, details_dir)


def plot_single_detail(res, rank, folder):
    
    tr_mse = res['tr_mse']
    tr_acc = res['tr_accuracy']
    vl_acc = res['vl_accuracy']
    epochs = range(1, len(tr_mse) + 1)
    
    best_val = max(vl_acc)
    best_ep = np.argmax(vl_acc) + 1
    
    # Figura larga
    # bottom=0.20 è sufficiente per il testo orizzontale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplots_adjust(bottom=0.20) 
    
    # Grafico 1: MSE
    ax1.plot(epochs, tr_mse, color='red', linewidth=2, label='Train MSE')
    ax1.set_title("Training Loss (MSE)", fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Grafico 2: Accuracy
    ax2.plot(epochs, tr_acc, color='red', label='Train Acc')
    ax2.plot(epochs, vl_acc, color='blue', linestyle='--', label='Val Acc')
    ax2.scatter(best_ep, best_val, color='green', s=100, marker='*', label='Best Val')
    
    ax2.set_title(f"Accuracy (Best: {best_val*100:.2f}%)", fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Testo Parametri (Layout ORIZZONTALE)
    params_txt = format_params(res['params'], layout='horizontal')
    
    info_box = (
        f"MODELLO #{rank}  |  Best Val Acc: {best_val*100:.2f}% (Epoca {best_ep})  |  MSE Finale: {tr_mse[-1]:.5f}\n"
        f"----------------------------------------------------------------------------------------------------\n"
        f"{params_txt}"
    )
    
    # Inseriamo il testo in basso al centro
    plt.figtext(0.5, 0.03, info_box, ha="center", va="bottom", fontsize=10, family='monospace',
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))
    
    filename = f"Rank_{rank}_Detail.png"
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def format_params(params, layout='horizontal'):
    # Parametri da ignorare
    ignore = ['verbose', 'validation', 'split', 'early_stopping', 'epsilon', 'n_outputs']
    
    parts = []
    for k, v in params.items():
        if k in ignore: continue
        
        # Abbreviazioni
        label = k
        if "mini_batch_size" in k: label = "Batch"
        elif "learning_rate" in k: label = "LR"
        elif "decay_factor" in k: label = "Decay"
        elif "decay_step" in k: label = "DecayStep"
        elif "momentum" in k: label = "Mom"
        elif "alpha" in k: label = "Alpha"
        elif "lambda" in k: label = "L2"
        elif "units" in k: label = "Units"
        elif "act" in k: label = "Act"
        
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__
        
        parts.append(f"{label}={val}")
    
    # Uniamo tutto in una stringa unica separata da virgole
    full_string = ", ".join(parts)
    
    if layout == 'grid':
        # Per la griglia piccola, andiamo a capo spesso (stretto)
        return "\n".join(textwrap.wrap(full_string, width=50))
    else:
        # Per il dettaglio orizzontale, usiamo una larghezza ampia (es. 110 caratteri)
        # così sembra un paragrafo e non una lista
        return "\n".join(textwrap.wrap(full_string, width=110))