import numpy as np
from src.training.trainer.trainer_cup import TrainerCup
from src.training.validation.hold_out import hold_out_validation
from config import cup_config
from src.training.grid_search import GridSearch
from src.activationf.relu import relu
from src.activationf.leaky_relu import leaky_relu
from src.activationf.linear import linear
from src.utils import load_data, normalize_data
from src.utils.compute_error import mean_euclidean_error_with_denorm, mean_squared_error_with_denorm

from src.training.validation.k_fold import run_k_fold_cup


def print_config(config, score=None, metric_name="MSE"):
    """Stampa helper per visualizzare bene i risultati"""
    print("\n" + "═"*60)
    print(f" 🏆  BEST CONFIGURATION FOUND")
    print("═"*60)
    if score is not None:
        print(f" 📊  BEST {metric_name:<20}: {score:.6f}")
        print("─"*60)
    for key in sorted(config.keys()):
        val = config[key]
        val_str = val.__name__ if hasattr(val, '__name__') else str(val)
        print(f" • {key:<25} :  {val_str}")
    print("═"*60 + "\n")

# =============================================================================
# 4. PLOTTING FINAL ASSESSMENT
# =============================================================================
# Poiché run_k_fold_cup restituisce le curve medie, possiamo plottarle direttamente.
# Importiamo la funzione di plotting se non l'abbiamo già fatto, o ne creiamo una ad hoc.
# Supponiamo di voler usare una funzione simile a plot_grid_analysis ma per un singolo risultato.

import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_final_assessment(results, avg_target_range, relative_path="results/cup/final_assessment"):
    """
    Plotta le curve medie di MSE e MEE (se disponibili) del Final Assessment K-Fold.
    """
    
    # Crea directory se non esiste
    # (Assumendo che tu abbia una funzione _ensure_dir o simile, altrimenti usa os.makedirs)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Adjust path as needed based on where run_cup.py is
    full_path = os.path.join(base_dir, relative_path)
    os.makedirs(full_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Estrai le curve
    tr_mse = results.get('mean_tr_history', [])
    vl_mse = results.get('mean_vl_history', [])
    
    # Se hai anche le curve MEE nel dizionario results (dipende da come hai implementato run_k_fold_cup)
    # Altrimenti plotta solo MSE. 
    # Assumiamo per ora solo MSE dato il return di run_k_fold_cup standard.
    
    epochs = np.arange(1, len(tr_mse) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tr_mse, label='Mean Train MSE', color='red')
    plt.plot(epochs, vl_mse, label='Mean Val MSE', color='blue', linestyle='--')
    
    # Denormalizzazione per label asse Y (opzionale, o plotta normalizzato)
    # plt.ylabel('MSE (Normalized)')
    
    plt.title(f"Final K-Fold Assessment - Mean Learning Curves\n(K={cup_config.FOLDS})")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.yscale('log') # MSE spesso si vede meglio in scala logaritmica
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    filename = f"Final_KFold_Curves_{timestamp}.png"
    plt.savefig(os.path.join(full_path, filename), dpi=300)
    print(f"\n✅ Grafico Final Assessment salvato in: {os.path.join(full_path, filename)}")
    plt.close()



# =============================================================================
# CARICAMENTO
# =============================================================================
print("📥 Caricamento dati CUP...")


x_i, d = load_data(cup_config.PATH_DT)
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

# Normalizza
x_i, x_min, x_max = normalize_data(x_i)
d, d_min, d_max = normalize_data(d)

x_i, d, x_i_test, d_test = hold_out_validation(x_i, d, 15) 

target_range = d_max - d_min
avg_target_range = np.mean(target_range)

print(f"Data shape: {x_i.shape}")
print(f"Target Range medio (per denormalizzazione): {avg_target_range:.4f}")

# =============================================================================
# GRID SEARCH 
# =============================================================================
gs = GridSearch(
    units_list=[
      #[128, 64, 32], 
      [100, 50]
    ],
    n_outputs=[cup_config.N_OUTPUTS],
    f_act_hidden=[leaky_relu],
    f_act_output=[linear],
    
    mini_batch_size=[32], 
    
    learning_rate=[
                0.0001, 
                #0.000025
                   ], 
    
    use_decay=[False],
    decay_factor=[0.9],
    decay_step=[25], 
    
    momentum=[0.9],
    alpha_mom=[
        0.6, 
        #0.2
        ],
    
    lambdal2=[0.0, 
              #0.001
              ], 
    
    epochs=[3000], 
    early_stopping=[True],
    epsilon=[1e-24],
    patience=[50],
    max_gradient_norm=[100],
    
    split=[cup_config.SPLIT],
    verbose=[False], 
    validation=[True]
)

print("\n🚀 Avvio Grid Search con K-Fold interno (Model Selection)...")

best_config, best_score_gs = gs.run_for_cup_with_kfold(x_i, d, k_folds=3)

print_config(best_config, best_score_gs, "Mean MSE (Grid)")


# =============================================================================
# 3. FINAL ASSESSMENT (K-Fold Intenso sul Best Model)
# =============================================================================
print("\n" + "█"*60)
print(f"🔎 AVVIO FINAL ASSESSMENT (Intense K-Fold)")
print("█"*60)

final_config = best_config.copy()
final_config['epochs'] = 5000      
final_config['patience'] = 50    
final_config['verbose'] = False   

k_folds_final = cup_config.FOLDS 

final_results = run_k_fold_cup(
    x_full=x_i,
    d_full=d,
    k_folds=k_folds_final,
    model_config=final_config,
    x_test_internal=x_i_test,
    verbose=True
)

"""final_result_test_internal = np.asarray(final_results["test_internal_history_output"])

print("\n\n\[DEBUG] final_results", final_result_test_internal.shape)
print(d_test.shape)

mee_test = []
mse_test = []

# d_test: 75x4, final_result: 3 x 4 
for i in range(final_result_test_internal.shape[0]):

    mse_test.append(mean_squared_error_with_denorm(final_result_test_internal[i], d_test, x_min, x_max, d_max, d_min))
    mee_test.append(mean_euclidean_error_with_denorm(final_result_test_internal[i], d_test, x_min, x_max, d_max, d_min))

print("||🙏 MEE_TEST_INTERNAL: ", mee_test)
print("||🙏 MSE_TEST_INTERNAL: ", mse_test)"""



#print("||| ✅ Final test internal result array mse: ", final_results["test_internal_history_mse"])

plot_final_assessment(final_results, avg_target_range)

# =============================================================================
# REPORT
# ================================
mean_mse_norm = final_results['mean_mse']
mean_mee_norm = final_results['mean_mee']

# Calcolo errori Denormalizzati (Reali)
# MSE scala col quadrato del range, MEE scala linearmente

mean_mse_real = mean_mse_norm * (avg_target_range ** 2) 
mean_mee_real = mean_mee_norm * avg_target_range



print("\n" + "="*60)
print(f"📄 REPORT FINALE (Media su {k_folds_final} Folds)")
print("="*60)

print(f"📊 Metriche NORMALIZZATE (0-1):")
print(f"   • Mean MSE: {mean_mse_norm:.6f} (± {final_results['std_mse']:.6f})")
print(f"   • Mean MEE: {mean_mee_norm:.6f}")
print("-" * 60)

print(f"🌍 Metriche REALI (Denormalizzate):")
print(f"   • Mean MSE: {mean_mse_real:.6f}")
print(f"   • Mean MEE: {mean_mee_real:.6f}") 
print("="*60)
print(f"Miglior config utilizzata:\n{final_config}")



