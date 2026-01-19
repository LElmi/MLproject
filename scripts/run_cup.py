from src.training.trainer.trainer import Trainer
import numpy as np
from src.training.validation.hold_out import hold_out_validation
#from scripts.run_validation import *
from config import cup_config
from src.training.grid_search import GridSearch
from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf
from src.activationf.linear import linear
from src.utils import *
from src.training.validation.k_fold import k_fold

# Carica dati

x_i, d = load_data(cup_config.PATH_DT)

x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)


# Normalizza Input
x_i, x_min, x_max = normalize_data(x_i)
d, d_min, d_max = normalize_data(d)

# --- HOLD OUT SPLIT --
tr_input, tr_target, vl_input, vl_target = hold_out_validation(x_i, d, cup_config.SPLIT)

# GridSearch è una classe che usa **kwargs come argomento, ergo,
# non ha limiti di argomenti e combinazioni,
# al suo interno usa la classe TRAIN! Da tenere in considerazione se si
# apportano modifiche lì!
gs = GridSearch(
    units_list=[[32,64], [32, 64, 32]],
    n_outputs = [cup_config.N_OUTPUTS],
    f_act_hidden = [relu],
    f_act_output = [linear],
    learning_rate = [0.0005,0.001,0.0001],
    use_decay = [True],
    decay_factor =[0.95] ,#if use_decay==True else [0.0],
    decay_step = [10],
    batch = [cup_config.BATCH],
    epochs=[1000],
    early_stopping = [True],
    epsilon = [1e-4],
    patience = [cup_config.PATIENCE],
    momentum = [cup_config.MOMENTUM],
    alpha_mom = [[0.9]],
    max_gradient_norm = [20],
    split = [cup_config.SPLIT],
    verbose = [True],
    validation = [True],
    lambdal2 = [1e-5, 1e-4, 1e-3]

)

best_config, best_mee_GS = gs.run_for_cup(tr_input, tr_target, vl_input, vl_target,mean_euclidean_error)

number_of_patterns_in_one_fold=round(x_i.shape[0]/cup_config.FOLDS)
mee_arr=np.zeros(cup_config.FOLDS)
best_mee_K_FOLD=9999.
best_accuracy=0.
accuracy_allmodels=np.zeros(cup_config.FOLDS)
trainers=[]
for validation_fold in range(cup_config.FOLDS):
    print("\nTraining fold ",validation_fold,"...\n")
    tr_input,tr_target,vl_input,vl_targets=k_fold(x_i, d,cup_config.FOLDS,validation_fold)
    if (cup_config.EARLY_STOPPING == True):
        trainers.append(
            Trainer(input_size=tr_input.shape[1], **best_config)
        )
        mee_tr, mse_tr, mee_vl, mse_vl, accuracy = trainers[validation_fold].fit_k_fold(tr_input, tr_target,validation_fold,
                                                    vl_input, vl_targets, mean_euclidean_error)
        mee_arr[validation_fold] = mee_vl
        #print("\n \n \n",accuracy_history)
        accuracy_allmodels[validation_fold] = accuracy

for i in range(mee_arr.shape[0]):
    if (mee_arr[i] < best_mee_K_FOLD ):
        print("entra")
        best_mee_K_FOLD = mee_arr[i]

print(" 🏆🚀 BEST CONFIG IN GRID SEARCH: \n", best_config, "\n", "BEST MEE IN GRID SEARCH: ", best_mee_GS)

print("\nMiglior modello trovato tra quelli analizzati con la K fold cross validation con ",cup_config.FOLDS," folds: \nmiglior mee=",best_mee_K_FOLD,"\ncon media:",np.mean(mee_arr), "\nl'array di tutte le folds", mee_arr)
