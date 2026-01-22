import numpy as np
from config import cup_config
from src.utils import * 
from src.activationf import *
from src.training.grid_search import GridSearch
from src.training.trainer.trainer_cup import TrainerCup
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
            #[2400, 1200],
            [1200, 600],
            [800, 400],
            [600, 300]
            #[200, 100],
            #[256, 128, 64]
            #[40,20]x
            ],

            n_outputs=[cup_config.N_OUTPUTS],
            f_act_hidden=[leaky_relu],
            f_act_output=[linear],


            mini_batch_size=[
                            32, #mmmm
                            64,
                            #16
                            ], 

            learning_rate=[
                        #0.01, 
                        #0.0025,
                        #0.008,
                        #0.015,
                        #0.01.  #mmm
                        0.02,
                        0.001,
                        0.005
                        ], 

            use_decay=[False],
            decay_factor=[0.9],
            decay_step=[
                #25,
                50
                ], 
            
            momentum=[True],
            alpha_mom=[
                #0.0,
                #0.5,
                0.9, #mmm
                #0.7
                ],
            
            lambdal2=[
                    0.00001, 
                    #0.001.  #mmm
                    ], 
    
            epochs=[5000], 
            early_stopping=[True],
            epsilon=[1e-20],
            patience=[70],
            max_gradient_norm=[100],
            
            split=[cup_config.SPLIT],
            verbose=[False], 
            validation=[True]
)


print("\n🚀 Avvio Grid Search con K-Fold interno (Model Selection)...")

k_folds = 3
best_config, best_score_gs, best_epoch, tr_history_error, vl_history_error = gs.run_for_cup_with_kfold(
    x_i, d, k_folds,
    d_max, d_min, # <- Parametri output denormalizzaione per i plot
    d_max, d_min  # <- Parametri target ""   ""  ""  
    )  # <- METRICHE TR ( = TR) E VL

print_config(best_config, best_score_gs, "Mean MSE (Grid)")


# =============================================================================
# 3. FINAL ASSESSMENT (K-Fold Intenso sul Best Model)
# =============================================================================
print("\n" + "█"*60)
print(f"🔎 AVVIO FINAL ASSESSMENT (Intense K-Fold)")
print("█"*60)


# Numero di K-FOLD
#k_folds_final = 3


#[ TOP 5 config ]

# Cambio configurazione per Train intenso
final_config = best_config.copy()
final_config['epochs'] = best_epoch  # Deve essere la migliore
final_config['early_stopping'] = False
final_config['validation'] = False
final_config['verbose'] = True   

# Istanza del Trainer, che instanzia una rete neurale in base ai parametri passati
trainer_final = TrainerCup(
    input_size = x_i.shape[1],
    **final_config
)

# Addestramento della NN
tr_history_error, _ = trainer_final.fit(x_i, d)                                   # <- METRICHE DEL TR ( = TR + VL)
# Run della NN dopo il fit
final_out = trainer_final.neuraln.run_nn(x_i_test)
# Denormalizzaione degli output
mee_final_test_denorm = denorm_mean_euclidean_error_test(          # <- METRICHE DEL TEST SET INTERNO
    final_out,    # Output rete
    d_test,       # Target rete
    d_max, d_min, # Range per denormalizzare Output (CORRETTO)
    d_max, d_min  # Range per denormalizzare Target
)









#mse_final_test_denorm = mean_euclidean_error_test(final_output_on_test_set, d_test, x_max, x_min, d_max, d_min)

print("\n\n|||| 🎬 MEE FINAL: ", mee_final_test_denorm, "|||\n\n")
