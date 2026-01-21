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
            #[400, 200],
            [1200, 600]
            #[200, 100],
            #[256, 128, 64]
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
                        #0.0025,
                        #0.008,
                        #0.015,
                        0.01
                        #0.001,
                        #0.0005
                        ], 

            use_decay=[False],
            decay_factor=[0.9],
            decay_step=[
                #25,
                50
                ], 
            
            momentum=[True],
            alpha_mom=[
                0.6, 
                #0.7
                ],
            
            lambdal2=[0.00001, 
                    #0.001
                    ], 
    
    epochs=[5000], 
    early_stopping=[True],
    epsilon=[1e-6],
    patience=[30],
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
# 4. FINAL INTEGRATED REPORT (Validation + Blind Test)
# =============================================================================

# --- 1. Calcolo Metriche su Blind Test (se disponibile) ---
# Usiamo la denormalizzazione VETTORIALE (esatta)
test_mse_real_mean, test_mse_real_std = 0.0, 0.0
test_mee_real_mean, test_mee_real_std = 0.0, 0.0
has_test_results = False

if "test_internal_history_output" in final_results:
    has_test_results = True
    final_result_test_internal = np.asarray(final_results["test_internal_history_output"])
    
    # Liste per accumulare gli errori di ogni modello (fold) sul test set
    test_mse_list = []
    test_mee_list = []

    for i in range(final_result_test_internal.shape[0]):
        # Calcolo MSE reale vettoriale
        mse_val = mean_squared_error_with_denorm(
            outputs=final_result_test_internal[i], 
            targets=d_test, 
            d_min=d_min, 
            d_max=d_max
        )
        test_mse_list.append(mse_val)

        # Calcolo MEE reale vettoriale
        mee_val = mean_euclidean_error_with_denorm(
            outputs=final_result_test_internal[i], 
            targets=d_test, 
            d_min=d_min, 
            d_max=d_max
        )
        test_mee_list.append(mee_val)
    
    # Medie e Deviazioni Standard
    test_mse_real_mean = np.mean(test_mse_list)
    test_mse_real_std = np.std(test_mse_list)
    test_mee_real_mean = np.mean(test_mee_list)
    test_mee_real_std = np.std(test_mee_list)

# --- 2. Calcolo Metriche su Validation (K-Fold) ---
# Usiamo la stima tramite AVG_TARGET_RANGE (approssimazione sui dati aggregati)
val_mse_norm = final_results['mean_mse']
val_mee_norm = final_results['mean_mee']
val_mse_std_norm = final_results['std_mse']
# (Nota: std_mee potrebbe non essere presente se non l'hai aggiunto in k-fold, metto 0 default)
val_mee_std_norm = final_results.get('std_mee', 0.0)

# Denormalizzazione Metriche Validation
# MSE scala col quadrato del range medio, MEE linearmente
val_mse_real_mean = val_mse_norm * (avg_target_range ** 2)
val_mse_real_std  = val_mse_std_norm * (avg_target_range ** 2)
val_mee_real_mean = val_mee_norm * avg_target_range
val_mee_real_std  = val_mee_std_norm * avg_target_range

# --- 3. STAMPA REPORT UNIFICATO ---
print("\n" + "═"*75)
print(f"📄 FINAL REPORT & ASSESSMENT (su {k_folds_final} Folds)")
print("═"*75)

print(f"{'METRICA':<10} | {'SET':<15} | {'VALORE REALE (Denorm)':<25} | {'VALORE NORM (0-1)':<20}")
print("-" * 75)

# MSE ROW
print(f"{'MSE':<10} | {'Validation':<15} | {val_mse_real_mean:.5f} ± {val_mse_real_std:.5f}      | {val_mse_norm:.5f}")
if has_test_results:
    print(f"{'':<10} | {'Blind Test':<15} | {test_mse_real_mean:.5f} ± {test_mse_real_std:.5f}      | {'-':<20}")

print("-" * 75)

# MEE ROW
print(f"{'MEE':<10} | {'Validation':<15} | {val_mee_real_mean:.5f} ± {val_mee_real_std:.5f}      | {val_mee_norm:.5f}")
if has_test_results:
    print(f"{'':<10} | {'Blind Test':<15} | {test_mee_real_mean:.5f} ± {test_mee_real_std:.5f}      | {'-':<20}")

print("═"*75)
print("💡 NOTE:")
print(" • Validation: Media delle performance stimate durante il K-Fold (Approssimazione tramite range medio).")
print(" • Blind Test: Media delle performance reali calcolate vettorialmente sui K modelli (Ensemble).")
print("═"*75)
print(f"Miglior config utilizzata:\n{final_config}")

# Plotting finale
plot_final_assessment(final_results, avg_target_range, k_folds_final)



"""# =============================================================================
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


"""
