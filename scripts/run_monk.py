import numpy as np
import itertools
import config
from src.training.train.trainer import Trainer
from src.utils import *

# Carica dati
x_i, d = load_data(config.PATH_DT)
x_i = x_i.to_numpy()
d = d.to_numpy()

x_i_remaining, validation_i = traindata_split_in_ts_vs(x_i, config.SPLIT)
x_i_remaining, x_min, x_max = normalize_data(x_i_remaining)

x_i_remaining, x_min, x_max = normalize_data(x_i)

denom = x_max - x_min
denom[denom == 0] = 1.0
validation_i = (validation_i - x_min) / denom
d = normalize_data(d)

# Parametri da esplorare in grid search

learning_rate_array = np.logspace(-5, -3, 10)
alpha_array = np.linspace(0.25, 0.75, 10)

# Fa il prodotto cartesiano dei valori, lo scorre nel ciclo for
hyperparams_combinations = list(itertools.product(learning_rate_array, alpha_array))

scouting_epochs = int(round(config.EPOCHS*0.05)) #5% delle epoche viene usato per ogni run della grid search
best_mee = float('inf')
best_config = None

print("----------------------------------------------------------------")
print(f"INIZIO GRID SEARCH")
print(f"Configurazioni totali: {len(hyperparams_combinations)}")
print(f"Epoche di scouting per config: {scouting_epochs}")
print("----------------------------------------------------------------")

# ============================================================
# ESECUZIONE GRID SEARCH
# ============================================================

# Scorre la lista di combinazioni esplorate
for idx, (lr, alpha) in enumerate(hyperparams_combinations):
    
    print(f"Esplorazione {idx+1}/{len(hyperparams_combinations)} -> LR: {lr:.5f}, Alpha: {alpha:.2f}", end="")

    scout_trainer = Trainer(
            input_size = x_i_remaining.shape[1],
            units_list = config.UNITS_LIST,
            n_outputs = config.N_OUTPUTS,
            f_act = config.FUN_ACT,
            learning_rate = lr,
            batch = config.BATCH,         
            epochs = scouting_epochs,     
            early_stopping = True,        
            epsilon = config.EPSILON * 10.,
            patience = config.PATIENCE,
            momentum = config.MOMENTUM,
            alpha_mom = alpha,            
            split = config.SPLIT
        )

    final_mee = scout_trainer.fit(x_i_remaining, d)
    print(f"MEE: {final_mee:.5f}")

    if final_mee<best_mee:
        best_mee=final_mee
        best_config = (lr, alpha)
        print(f"   >>> NUOVO RECORD! LR={lr:.5f}, Alpha={alpha:.2f}")


# ============================================================
# TRAINING FINALE (BEST MODEL)
# ============================================================

best_lr, best_alpha = best_config

print("----------------------------------------------------------------")
print("GRID SEARCH TERMINATA")
print(f"Miglior Configurazione: LR={best_lr}, Alpha={best_alpha}")
print(f"Miglior MEE Scouting: {best_mee}")
print("Avvio Training Finale...")
print("----------------------------------------------------------------")

# Istanziamo il Trainer finale
final_trainer = Trainer(
        input_size=x_i_remaining.shape[1],
        n_hidden1=config.N_HIDDENL1,
        n_hidden2=config.N_HIDDENL2,
        n_outputs=config.N_OUTPUTS,
        f_act=config.FUN_ACT,
        learning_rate=best_lr,          
        batch=config.BATCH,
        epochs=config.EPOCHS,          
        early_stopping=config.EARLY_STOPPING, 
        epsilon=config.EPSILON,         
        patience=config.PATIENCE,
        momentum=config.MOMENTUM,
        alpha_mom=best_alpha,           
        split=config.SPLIT,
        verbose=True
    )

# Avvia il training (verbose=True per vedere i progressi)
final_trainer.fit(x_i_remaining, d)

"""