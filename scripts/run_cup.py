import numpy as np
from config import cup_config
from src.utils import * 
from src.activationf import *
from src.training.grid_search import GridSearch
from src.training.validation.k_fold import run_k_fold_cup
from src.training.validation.hold_out import hold_out_validation


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
      [100, 50],
      [200, 100],
      [124, 64]
    ],
    n_outputs=[cup_config.N_OUTPUTS],
    f_act_hidden=[leaky_relu],
    f_act_output=[linear],
    
    mini_batch_size=[
                    #32, 
                    16
                     ], 
    
    learning_rate=[
                #0.01, 
                0.0025,
                0.005
                #0.001,
                #0.0005
                   ], 
    
    use_decay=[True],
    decay_factor=[0.9],
    decay_step=[10], 
    
    momentum=[False],
    alpha_mom=[
        0.6, 
        #0.7
        ],
    
    lambdal2=[0.00001, 
              #0.001
              ], 
    
    epochs=[5000], 
    early_stopping=[True],
    epsilon=[1e-20],
    patience=[500],
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

# Cambio configurazione per Train intenso
final_config = best_config.copy()
final_config['epochs'] = 5000      
final_config['patience'] = 500   
final_config['epsilon'] = 1e-20
final_config['verbose'] = False   

# Numero di K-FOLD
k_folds_final = 3

final_results = run_k_fold_cup(
    x_full=x_i,
    d_full=d,
    k_folds=k_folds_final,
    model_config=final_config,
    x_test_internal=x_i_test,
    verbose=True
)

# =============================================================================
# FINAL TEST INTERNAL
# =============================================================================

if "test_internal_history_output" in final_results:
    final_result_test_internal = np.asarray(final_results["test_internal_history_output"])
    
    print(f"\n[DEBUG] Shape Test Results: {final_result_test_internal.shape} (K_Folds, N_Test, Outputs)")
    
    mee_test_list = []
    mse_test_list = []

    # Iteriamo su ogni fold (i)
    # d_test è il target del test set (normalizzato)
    for i in range(final_result_test_internal.shape[0]):
        
        # Calcolo MSE reale
        mse_val = mean_squared_error_with_denorm(
            outputs=final_result_test_internal[i], 
            targets=d_test, 
            d_min=d_min, 
            d_max=d_max
        )
        mse_test_list.append(mse_val)

        # Calcolo MEE reale
        mee_val = mean_euclidean_error_with_denorm(
            outputs=final_result_test_internal[i], 
            targets=d_test, 
            d_min=d_min, 
            d_max=d_max
        )
        mee_test_list.append(mee_val)

    print("\n" + "═"*60)
    print("🌍 BLIND TEST INTERNO (Denormalizzato)")
    print("═"*60)
    print(f"Mean MSE (Real): {np.mean(mse_test_list):.5f} (± {np.std(mse_test_list):.5f})")
    print(f"Mean MEE (Real): {np.mean(mee_test_list):.5f} (± {np.std(mee_test_list):.5f})")
    print("═"*60)

#print("||| ✅ Final test internal result array mse: ", final_results["test_internal_history_mse"])

plot_final_assessment(final_results, avg_target_range, k_folds_final)

# =============================================================================
# REPORT
# =============================================================================
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



