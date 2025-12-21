from src.training import trainer
from src.training.trainer import Trainer
import numpy as np
from src.utils.load_data import load_data

import config

# Carica dati
x_i, d = load_data(config.PATH_DT)
x_i = x_i.to_numpy()

#-----------------------------------------------------------------
#---hold out una percentuale uguale a config.SPLIT/100 es. 20/100
#-----------------------------------------------------------------
n_total=int(x_i.shape[0])
n_keep=int(round(n_total-n_total*config.SPLIT/100.))
validation_i=x_i[n_keep:,:]                                       #<- i rimanenti valori vengono caricati in una array per la validation
x_i_remaining = x_i[:n_keep,:]
#-----------------------------------------------------------------
d = d.to_numpy()

# Inizializza la classe Trainer:
#  - crea la rete neurale
#  - traccia gli errori
#  - implementa metodo che esegue il training



# normalizza le matrici x_i e d in l2
min_val_x = x_i_remaining.min(axis=0)
max_val_x = x_i_remaining.max(axis=0)
x_i_remaining = (x_i_remaining - min_val_x) / (max_val_x - min_val_x)

min_val_d = d.min(axis=0)
max_val_d = d.max(axis=0)
d = (d - min_val_d) / (max_val_d - min_val_d)

#------------------------Grid search--------------------------#
perc=np.array([-0.50,-0.40,-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50]) #<--- usando il valore in config.py come valore centrale
learning_rate_array=config.LEARNING_RATE * (1+perc)                     #     creo un vettore aumentando/diminuendo tale valore di una data percentuale
alpha_array=config.ALPHA_MOM * (1+perc)
scouting_epochs=int(round(config.EPOCHS*0.03)) #3% delle epoche viene usato per ogni run della grid search
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
                              config.SPLIT) for i in range(num_cols)]
scout = [list(row) for i in range(num_rows)]
#--------------------------------------
# ciclo su lo spazio dei parametri per la grid search
#--------------------------------------
print("-------------------------------------------------------------------------------------------------------------")
print("inizio scouting per ",scouting_epochs," epochs per ogni configurazione, il numero di configurazioni possibili è: ",perc.shape[0]**2, " per un totale di ",perc.shape[0]**2 * scouting_epochs," epochs")
print("I range di parametri che verranno testati sono:")
print("learning rate range = ",config.LEARNING_RATE * (1+min(perc)),"-",config.LEARNING_RATE * (1+max(perc)))
print("Alpha momentum range = ",config.ALPHA_MOM * (1+min(perc)),"-",config.ALPHA_MOM * (1+max(perc)))
print("-------------------------------------------------------------------------------------------------------------")
for i in range(perc.shape[0]):
    for j in range(perc.shape[0]):
        #for k in range(perc.shape[0]):
            scout[i][j] = Trainer(x_i_remaining.shape[1],
                              config.N_HIDDENL1,
                              config.N_HIDDENL2,
                              config.N_OUTPUTS,
                              config.FUN_ACT,
                              learning_rate_array[i],
                              config.BATCH,
                              scouting_epochs,
                              config.EARLY_STOPPING,
                              config.EPSILON,
                              config.PATIENCE,
                              config.MOMENTUM,
                              alpha_array[j],
                              config.SPLIT)
            mee=scout[i][j].grid_search_train(x_i_remaining,d, scouting_epochs)
            print("scouting con parametri: learning rate=", learning_rate_array[i], ", alpha=", alpha_array[j])
            if (mee<best_mee):
                best_mee=mee
                best_index_alpha=j
                best_index_learning_rate=i
                print("Miglior MEE aggiornata a:", best_mee)
print("-------------------------------------------------------------------------------------------------------------")
print("SCOUTING TERMINATO")
print("Miglior MEE trovata: MEE=",best_mee)
print("Parametri corrispondenti: learning rate = ",learning_rate_array[best_index_learning_rate],"alpha = ",alpha_array[best_index_alpha])#," epsilon = ",epsilon_array[best_index_epsilon],])
print("-------------------------------------------------------------------------------------------------------------")
print("INIZIO TRAINING")
# Avvia training
trainer = Trainer(x_i_remaining.shape[1],
                      config.N_HIDDENL1,
                      config.N_HIDDENL2,
                      config.N_OUTPUTS,
                      config.FUN_ACT,
                      learning_rate_array[best_index_learning_rate],
                      config.BATCH,
                      config.EPOCHS,
                      config.EARLY_STOPPING,
                      config.EPSILON,
                      config.PATIENCE,
                      config.MOMENTUM,
                      alpha_array[best_index_alpha],
                      config.SPLIT)
if (config.EARLY_STOPPING == True):
    trainer.train_with_early_stopping(x_i_remaining, d)
else:
    trainer.train_standard(x_i_remaining, d)
