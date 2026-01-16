from src.training.trainer import trainer
from src.training.trainer.trainer import Trainer
from src.training.grid_search import GridSearch
import numpy as np
from src.utils.load_data import load_monks_data
#from scripts.run_validation import *
from src.training.validation.hold_out import hold_out_validation
from src.training.validation.k_fold import k_fold
from src.utils.load_model import *
from src.activationf.relu import relu
from src.utils.normalize_data import normalize_data
from src.activationf.sigmoid import sigmaf
from src.activationf.leaky_relu import leaky_relu
from config import monk_config

# Carica dati

x_i, d = load_monks_data(monk_config.PATH_DT)

x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

x_i, x_min, x_max = normalize_data(x_i)




# GridSearch è una classe che usa **kwargs come argomento, ergo, 
# non ha limiti di argomenti e combinazioni, 
# al suo interno usa la classe TRAIN! Da tenere in considerazione se si
# apportano modifiche lì!
if monk_config.RUN_HOLD_OUT_VALIDATION:

    tr_input, tr_target, vl_input, vl_targets = hold_out_validation(x_i, d, monk_config.SPLIT)
gs = GridSearch(
    units_list = [[4], [8]],
    n_outputs = [monk_config.N_OUTPUTS],
    f_act=[relu, sigmaf, leaky_relu],
    learning_rate = [0.001, 0.01, 0.005],
    use_decay = [True, False],
    decay_factor =[0.99, 0.95] ,#if use_decay==True else [0.0],
    decay_step = [100],
    batch = [monk_config.BATCH],
    #epochs=[100],
    early_stopping = [monk_config.EARLY_STOPPING],
    epsilon = [monk_config.EPSILON],
    patience = [monk_config.PATIENCE],
    momentum = [monk_config.MOMENTUM],
    alpha_mom = [[0.9],[0.5]],
    split = [monk_config.SPLIT],
    verbose = [monk_config.VERBOSE],
    validation = [monk_config.RUN_HOLD_OUT_VALIDATION],
)
best_config, best_mee = gs.run(tr_input, tr_target,vl_input,vl_targets, scouting_epochs=50)




if monk_config.RUN_K_FOLD:
    number_of_patterns_in_one_fold=round(x_i.shape[0]/monk_config.FOLDS)
    mee_arr=np.zeros(monk_config.FOLDS)
    best_mee=9999.
    best_accuracy=0.
    accuracy_allmodels=np.zeros(monk_config.FOLDS)
    best_model= Trainer(x_i.shape[1],
                          monk_config.UNITS_LIST,
                          monk_config.N_OUTPUTS,
                          monk_config.FUN_ACT,
                          monk_config.LEARNING_RATE,
                          monk_config.USE_DECAY,
                          monk_config.DECAY_FACTOR,
                          monk_config.DECAY_STEP,
                          monk_config.BATCH,
                          monk_config.EPOCHS,
                          monk_config.EARLY_STOPPING,
                          monk_config.EPSILON,
                          monk_config.PATIENCE,
                          monk_config.MOMENTUM,
                          monk_config.ALPHA_MOM,
                          monk_config.SPLIT,
                          monk_config.VERBOSE,
                          monk_config.RUN_HOLD_OUT_VALIDATION)
    for validation_fold in range(monk_config.FOLDS):

        tr_input,tr_target,vl_input,vl_targets=k_fold(x_i, d,monk_config.FOLDS,validation_fold)

        trainer = Trainer(input_size=tr_input.shape[1],
                        epochs= monk_config.EPOCHS,
                        **best_config)
        if (monk_config.EARLY_STOPPING == True):
            mee_tr, mse_tr, mee_vl, mse_vl,accuracy = trainer.fit_K_fold(tr_input, tr_target,validation_fold,
                                                         vl_input, vl_targets)
            mee_arr[validation_fold] = mee_vl
            #print("\n \n \n",accuracy_history)
            accuracy_allmodels[validation_fold] = accuracy
        for i in range(mee_arr.shape[0]):
            if (best_accuracy < accuracy ):
                best_mee = mee_arr[i]
                best_accuracy=accuracy
                best_model=trainer
        mean_accuracy=np.mean(accuracy_allmodels)
print(" 🏆🚀 BEST CONFIG: \n", best_config, "\n\n\n", "BEST MEE: ", best_mee)
print("miglior modello trovato tra quelli analizzati con la K fold cross validation con ",monk_config.FOLDS," folds:",best_mee, " con accuracy:",best_accuracy*100.,"%", " e accuracy media:", mean_accuracy*100,"%")



# Avvia training normale, da aggiustare per farlo andare correttamente se non diamo vl_input e vl_targets alla classe Trainer
# ha senso runnare il training sul migliore modello trovato tramite la K-fold? chi lo sa

#mee_tr, mse_tr, mee_vl, mse_vl = trainer.fit(x_i, d)


#PER RUNNARE IL TEST SET

#x_i, d = load_monks_data(monk_config.TEST_PATH_DT)

'''
if (monk_config.EARLY_STOPPING == True):
    mee_tr, mse_tr, mee_vl, mse_vl =trainer.fit(tr_input, tr_target, 
                             vl_input, vl_targets)
else:


    trainer.fit(tr_input, tr_target, vl_input, vl_targets)'''

