from src.training import trainer
from src.training.trainer import Trainer
import numpy as np
from src.utils.load_data import load_data, load_monks_data
from scripts.run_validation import *
import config
from src.utils.load_model import *


# Carica dati

if config.MONK:
    x_i, d = load_monks_data(config.PATH_DT)
else:
    x_i, d = load_data(config.PATH_DT)

x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

if config.MONK:
    min_val_x = x_i.min(axis=0)
    max_val_x = x_i.max(axis=0)
    x_i = (x_i - min_val_x) / (max_val_x - min_val_x)

else:
    min_val_x = x_i.min(axis=0)
    max_val_x = x_i.max(axis=0)
    x_i = (x_i - min_val_x) / (max_val_x - min_val_x)

    min_val_d = d.min(axis=0)
    max_val_d = d.max(axis=0)
    d = (d - min_val_d) / (max_val_d - min_val_d)

# --- HOLD OUT SPLIT ---
if config.RUN_HOLD_OUT_VALIDATION:
    n_total = x_i.shape[0]
    n_keep = int(round(n_total - n_total * config.SPLIT / 100.0))

    x_i_remaining = x_i[:n_keep]
    d_remaining = d[:n_keep]

    validation_set = x_i[n_keep:]
    validation_d = d[n_keep:]
else:
    x_i_remaining = x_i
    d_remaining = d
    validation_set = None
    validation_d = None

#-----------------------------------------------------------------

# Inizializza la classe Trainer:
#  - crea la rete neurale
#  - traccia gli errori
#  - implementa metodo che esegue il training





#------------------------Grid search--------------------------#
perc=np.array([-0.50,-0.40,-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50]) #<--- usando il valore in config.py come valore centrale
#learning_rate_array=config.LEARNING_RATE * (1+perc)                     #     creo un vettore aumentando/diminuendo tale valore di una data percentuale
#alpha_array=config.ALPHA_MOM * (1+perc)
#---------------alternativamente--------------

learning_rate_array=np.logspace(-6, -4, 10)
alpha_array=np.linspace(0.25, 0.75, 10)


scouting_epochs=int(round(config.EPOCHS*0.02)) #2% delle epoche viene usato per ogni run della grid search
mee=0
best_mee=9999.
best_index_learning_rate=0
best_index_alpha=0
#----------------------------------------------------
# inizializza una tabella di oggetti Trainer
#----------------------------------------------------
num_rows = perc.shape[0]
num_cols = perc.shape[0]
row = [Trainer(x_i_remaining.shape[1],
                              config.N_HIDDENL1,
                              config.N_HIDDENL2,
                              config.N_OUTPUTS,
                              config.FUN_ACT,
                              config.LEARNING_RATE,
                              config.BATCH,
                              config.EPOCHS,
                              config.EARLY_STOPPING,
                              config.EPSILON,
                              config.PATIENCE,
                              config.MOMENTUM,
                              config.ALPHA_MOM,
                              config.SPLIT,
                              config.LAMBDA) for i in range(num_cols)]
scout = [list(row) for j in range(num_rows)]
'''
#--------------------------------------
# ciclo su lo spazio dei parametri per la grid search
#--------------------------------------
print("-------------------------------------------------------------------------------------------------------------")
print("inizio scouting per ",scouting_epochs," epochs per ogni configurazione, il numero di configurazioni possibili è: ",learning_rate_array.shape[0]*alpha_array.shape[0], " per un totale di ",learning_rate_array.shape[0]*alpha_array.shape[0] * scouting_epochs," epochs")
print("I range di parametri che verranno testati sono:")
print("learning rate range = ",min(learning_rate_array),"-",max(learning_rate_array))
print("Alpha momentum range = ",min(alpha_array),"-",max(alpha_array))

count=0
for i in range(learning_rate_array.shape[0]):
    for j in range(alpha_array.shape[0]):
        #for k in range(perc.shape[0]):
            count+=1
            print("----------------------------------------------------------------------------------------------------------------------------")
            scout[i][j] = Trainer(x_i_remaining.shape[1],
                              config.N_HIDDENL1,
                              config.N_HIDDENL2,
                              config.N_OUTPUTS,
                              config.FUN_ACT,
                              learning_rate_array[i],
                              config.BATCH,
                              scouting_epochs,
                              #config.EPOCHS,
                              config.EARLY_STOPPING,
                              config.EPSILON*30., #Early stopping aggressivo per la fase di grid search
                              config.PATIENCE,
                              config.MOMENTUM,
                              alpha_array[j],
                              config.SPLIT,
                              config.LAMBDA)
            print("Configurazione n°:", count, "/", learning_rate_array.shape[0] * alpha_array.shape[0],
                    " scouting con parametri: learning rate=", learning_rate_array[i], ", alpha=", alpha_array[j])

            mee=scout[i][j].grid_search_train_with_early_stopping(x_i_remaining,d, scouting_epochs)

            if (mee<best_mee):
                best_mee=mee
                best_index_alpha=j
                best_index_learning_rate=i
                print("Miglior MEE aggiornata a:", best_mee)
print("----------------------------------------------------------------------------------------------------------------------------")
print("SCOUTING TERMINATO")
print("Miglior MEE trovata: MEE=",best_mee)
print("Parametri corrispondenti: learning rate = ",learning_rate_array[best_index_learning_rate],"alpha = ",alpha_array[best_index_alpha])#," epsilon = ",epsilon_array[best_index_epsilon],])
print("----------------------------------------------------------------------------------------------------------------------------")
'''
# Avvia training
trainer = Trainer(x_i_remaining.shape[1],
                      config.N_HIDDENL1,
                      config.N_HIDDENL2,
                      config.N_OUTPUTS,
                      config.FUN_ACT,
                      #learning_rate_array[best_index_learning_rate],
                      config.LEARNING_RATE,
                      config.BATCH,
                      config.EPOCHS,
                      config.EARLY_STOPPING,
                      config.EPSILON,
                      config.PATIENCE,
                      config.MOMENTUM,
                      #alpha_array[best_index_alpha],
                      config.ALPHA_MOM,
                      config.SPLIT,
                      config.LAMBDA)
if (config.EARLY_STOPPING == True):
    weights_filename, architecture_filename=trainer.train_with_early_stopping(x_i_remaining, d, validation_set, validation_d)
else:
    trainer.train_standard(x_i_remaining, d)
if (config.RUN_HOLD_OUT_VALIDATION == True):
    w_j1i,w_j2j1, w_kj2,   architecture=load_model(weights_filename,architecture_filename)
    validation_monk(validation_set,validation_d,w_j1i, w_j2j1, w_kj2)
