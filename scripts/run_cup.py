import numpy as np
from src.training.trainer.trainer_cup import TrainerCup
from src.training.validation.hold_out import hold_out_validation
from config import cup_config
from src.training.grid_search import GridSearch
from src.activationf.relu import relu
from src.activationf.leaky_relu import leaky_relu
from src.activationf.linear import linear
from src.utils import load_data, normalize_data

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
# CARICAMENTO
# =============================================================================
print("📥 Caricamento dati CUP...")
x_i, d = load_data(cup_config.PATH_DT)
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

# Normalizza
x_i, x_min, x_max = normalize_data(x_i)
d, d_min, d_max = normalize_data(d)

target_range = d_max - d_min
avg_target_range = np.mean(target_range)

print(f"Data shape: {x_i.shape}")
print(f"Target Range medio (per denormalizzazione): {avg_target_range:.4f}")

# =============================================================================
# GRID SEARCH 
# =============================================================================
gs = GridSearch(
    units_list=[
      [128, 64, 32]   
    ],
    n_outputs=[cup_config.N_OUTPUTS],
    f_act_hidden=[leaky_relu],
    f_act_output=[linear],
    
    mini_batch_size=[32], 
    
    learning_rate=[0.01], 
    
    use_decay=[True],
    decay_factor=[0.9],
    decay_step=[25, 50], 
    
    momentum=[0.9],
    alpha_mom=[0.7, 0.9],
    
    lambdal2=[0.01, 0.001], 
    
    epochs=[3000], 
    early_stopping=[True],
    epsilon=[1e-12],
    patience=[100, 150],
    max_gradient_norm=[9999],
    
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
    verbose=True
)

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