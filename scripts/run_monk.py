import numpy as np
from config import monk_config
from src.training.trainer.trainer_monk import TrainerMonk
from src.training.grid_search import GridSearch
from src.activationf import *
from src.utils import *
from src.training.validation.stratified_split import hold_out_validation_stratified

# Carica dati
x_i, d = load_monks_data("data/monk/train_data/monks-3.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)


tr_input, tr_target, vl_input, vl_target = hold_out_validation_stratified(x_i, d, monk_config.SPLIT)


# GridSearch è una classe che usa **kwargs come argomento, ergo, 
# non ha limiti di argomenti e combinazioni, 
# al suo interno usa la classe TRAIN! Da tenere in considerazione se si
# apportano modifiche lì!

gs = GridSearch(

    units_list = [
        [3], #best
        #[2],
        #[4]
        #[2, 2],
        #[3, 3],
        ],  
    n_outputs = [monk_config.N_OUTPUTS],
    f_act_hidden = [sigmaf,
                     #relu
                     ], 
    f_act_output = [sigmaf],
    
    learning_rate = [
            0.25, 
            #0.1,    # best
            #0.05,
            #0.001,
            #0.01,
            #0.005
            #0.16
            ], 
    
    use_decay = [False],     
    decay_factor = [0.9],
    decay_step = [10],
    
    mini_batch_size = [
        len(tr_input),
        #1,
        #25,
        #50,    # best
        #len(tr_input) * 0.5,
        #len(tr_input) * 0.33,       
        #len(tr_input) * 0.2,

    ],
    
    epochs=[100],
    
    # --- EARLY STOPPING (Fondamentale) ---
    early_stopping = [True],
    patience = [
        #5,
        #10,
        15,
        #20,
        #25,
        #30,
        40
        #75,
        #100,
        #125,
        #150,
        #200,
        #250
        #250
        ],       # Stop se accuracy non migliora per 20 epoche
    epsilon = [
        1e-6,
        1e-7,
        1e-8,
        1e-9
        ],
    
    momentum = [True],  
    alpha_mom = [
                #0.0,
                #0.6,
                0.9   #best
                #0.95,
                #0.8,
                #0.75,
                #0.7,
                #0.5
                ],
    max_gradient_norm = [100],
    
    split = [monk_config.SPLIT],
    verbose = [False],    
    validation = [True],  
    lambdal2 = [
        #0.00001, 
        0,
        #0.0001,  #best
        #0.00001
        #0.00001
        ]
)

print("\n🚀 Avvio Grid Search...")
best_config, best_acc_gs = gs.run_for_monk_holdout(tr_input, tr_target, vl_input, vl_target)

print("\n" + "═"*60)
print(f"🏆 MIGLIOR CONFIGURAZIONE TROVATA (Val Acc: {best_acc_gs:.2%})")
print("═"*60)
for k, v in best_config.items():
    print(f" • {k:<20}: {v}")
print("═"*60)

# =============================================================================
# 3. FINAL RETRAINING & TEST (Assessment)
# =============================================================================
# Ora che abbiamo i parametri migliori, riaddestriamo su TUTTO il training set (tr + vl)
# e testiamo sul file di test separato (monks-X.test) se esiste.

print("\n🔁 Retraining finale sul dataset completo...")

# Carica il Test Set (se definito in config)
# Assumiamo tu abbia definito PATH_TS in monk_config
try:
    x_test, d_test = load_monks_data("data/monk/test_data/monks-3.test")
    x_test = x_test.to_numpy().astype(np.float64)
    d_test = d_test.to_numpy().astype(np.float64)
    has_test_set = True
    print(f"Test Set caricato: {x_test.shape[0]} patterns")
except:
    print("⚠️ Nessun Test Set trovato (PATH_TS). Uso la validazione come stima finale.")
    has_test_set = False


# Tolgo l'early 
final_config = best_config.copy()
final_config['early_stopping'] = False 

# Creiamo il trainer finale
final_trainer = TrainerMonk(input_size=x_test.shape[1],**final_config)
# Forziamo verbose=True per vedere il training
final_trainer.verbose = True 

# Addestriamo su x_i (tutto il dataset di train originale)
# Passiamo x_test come validazione solo per vedere i grafici, ma NON per fermarci (early stopping)
# Oppure usiamo early stopping su una piccola porzione se necessario.
# Per il Monk standard, spesso si fa training a epoche fisse o fino a MSE < tot.
mse_final, acc_final_tr = final_trainer.fit(x_i, d, 
                                            ts_x=x_test if has_test_set else None, 
                                            ts_d=d_test if has_test_set else None)

print("\n" + "█"*60)
print("📄 REPORT FINALE MONK")
print("█"*60)
print(f"MSE Finale Training: {mse_final:.5f}")
print(f"Accuracy Training:   {acc_final_tr:.2%}")

if has_test_set:
    # Calcoliamo Accuracy sul Test Set
    acc_test = final_trainer._compute_accuracy_internal(x_test, d_test)
    print(f"Accuracy TEST SET:   {acc_test:.2%}")

print("█"*60)