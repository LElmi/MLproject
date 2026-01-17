from src.training.trainer import trainer
from src.training.trainer.trainer import Trainer
from src.training.grid_search import GridSearch
import numpy as np
from src.training.validation.hold_out import hold_out_validation
from src.activationf.leaky_relu import leaky_relu
from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf
from src.utils.compute_accuracy import compute_accuracy
from config import monk_config
from src.utils import *
from src.training.validation.k_fold import k_fold

# Carica dati
x_i, d = load_monks_data(monk_config.PATH_DT)
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)
x_i, x_min, x_max = normalize_data(x_i)


if monk_config.RUN_HOLD_OUT_VALIDATION:

    tr_input, tr_target, vl_input, vl_target = hold_out_validation(x_i, d, monk_config.SPLIT)


# GridSearch è una classe che usa **kwargs come argomento, ergo, 
# non ha limiti di argomenti e combinazioni, 
# al suo interno usa la classe TRAIN! Da tenere in considerazione se si
# apportano modifiche lì!
"""gs = GridSearch(
    units_list = [[32, 16], [64, 64], [128, 64, 32]], 
    learning_rate = [0.001, 0.01],
    f_act = [relu, sigmaf],
    use_decay = [True, False],
    decay_factor = [0.99, 0.95],
    decay_step = [100],
    n_outputs = [4], 
    batch = [True],
    early_stopping = [True],mee
    epsilon = [1e-5],
    patience = [20],
    momentum = [True],
    alpha_mom = [0.9],
    split = [0.2]
)

best_config, best_mee = gs.run(tr_input, tr_target, scouting_epochs=100)
print(" 🏆🚀 BEST CONFIG: \n", best_config, "\n\n\n", "BEST MEE: ", best_mee)
"""


gs = GridSearch(
    units_list=[[16,8], [8, 4]],
    n_outputs = [monk_config.N_OUTPUTS],
    f_act_hidden = [leaky_relu],
    f_act_output = [sigmaf],
    learning_rate = [0.005],
    use_decay = [True],
    decay_factor =[0.90] ,#if use_decay==True else [0.0],
    decay_step = [10],
    batch = [monk_config.BATCH],
    epochs=[200],
    early_stopping = [False],
    epsilon = [monk_config.EPSILON],
    patience = [monk_config.PATIENCE],
    momentum = [monk_config.MOMENTUM],
    alpha_mom = [[0.9]],
    max_gradient_norm = [20],
    split = [monk_config.SPLIT],
    verbose = [True],
    validation = [True],
    lambdal2 = [1e-5, 1e-4, 1e-3]

)
best_config, best_accuracy_GS = gs.run_for_monk(tr_input, tr_target, vl_input, vl_target, compute_accuracy)

number_of_patterns_in_one_fold=round(x_i.shape[0]/monk_config.FOLDS)
mee_arr=np.zeros(monk_config.FOLDS)
best_mee=9999.
best_accuracy=0.
accuracy_allmodels=np.zeros(monk_config.FOLDS)

trainers=[]
for validation_fold in range(monk_config.FOLDS):
    print("\nTraining fold ",validation_fold,"...\n")
    tr_input,tr_target,vl_input,vl_targets=k_fold(x_i, d,monk_config.FOLDS,validation_fold)
    if (monk_config.EARLY_STOPPING == True):
        trainers.append(
            Trainer(input_size=tr_input.shape[1], **best_config)
        )
        mee_tr, mse_tr, mee_vl, mse_vl, accuracy = trainers[validation_fold].fit_k_fold(tr_input, tr_target,validation_fold,
                                                    vl_input, vl_targets, compute_accuracy)
        mee_arr[validation_fold] = mee_vl
        #print("\n \n \n",accuracy_history)
        accuracy_allmodels[validation_fold] = accuracy

    for i in range(mee_arr.shape[0]):
        if (best_accuracy < accuracy ):
            best_mee = mee_arr[i]
            best_accuracy=accuracy
            best_model=trainer
    mean_accuracy=np.mean(accuracy_allmodels)

print(" 🏆🚀 BEST CONFIG IN GRID SEARCH: \n", best_config, "\n\n\n", "BEST ACCURACY: ", best_accuracy_GS)

print("\n\n\n\nMiglior modello trovato tra quelli analizzati con la K fold cross validation con ",monk_config.FOLDS," folds: \nmiglior mee=",best_mee, "\n con accuracy=",best_accuracy*100.,"%", "\n e accuracy media=", mean_accuracy*100,"%\n\n\n",accuracy_allmodels)



