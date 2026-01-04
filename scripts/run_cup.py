from src.training.trainer.trainer import Trainer
import numpy as np
from src.training.validation.hold_out import hold_out_validation
#from scripts.run_validation import *
from config import cup_config
from src.training.grid_search import GridSearch
from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf
from src.utils import * 


# Carica dati

x_i, d = load_data(cup_config.PATH_DT)

x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)


# Normalizza Input
x_i, x_min, x_max = normalize_data(x_i)
d, d_min, d_max = normalize_data(d)

# --- HOLD OUT SPLIT ---
if cup_config.RUN_HOLD_OUT_VALIDATION:

    tr_input, tr_target, vl_input, vl_target = hold_out_validation(x_i, d, cup_config.SPLIT)

#elif config.K_FOLD:
# ...
else:
    x_i_remaining = x_i
    d_remaining = d
    validation_set = None
    validation_d = None

# GridSearch è una classe che usa **kwargs come argomento, ergo, 
# non ha limiti di argomenti e combinazioni, 
# al suo interno usa la classe TRAIN! Da tenere in considerazione se si
# apportano modifiche lì!
gs = GridSearch(
    units_list = [[32, 16], [64, 64], [128, 64, 32]], 
    learning_rate = [0.001, 0.01],
    f_act = [relu, sigmaf],
    use_decay = [True, False],
    decay_factor = [0.99, 0.95],
    decay_step = [100],
    n_outputs = [4], 
    batch = [True],
    early_stopping = [True],
    epsilon = [1e-5],
    patience = [20],
    momentum = [True],
    alpha_mom = [0.9],
    split = [0.2]
)

best_config, best_mee = gs.run(tr_input, tr_target, scouting_epochs=100)
print(" 🏆🚀 BEST CONFIG: \n", best_config, "\n\n\n", "BEST MEE: ", best_mee)

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
"""# Avvia training
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
    validation_monk(validation_set,validation_d,w_j1i, w_j2j1, w_kj2)"""