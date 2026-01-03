from src.training.trainer import trainer
from src.training.trainer.trainer import Trainer
from src.training.grid_search import GridSearch
import numpy as np
from src.utils.load_data import load_monks_data
#from scripts.run_validation import *
from src.training.validation.hold_out import hold_out_validation
from src.utils.load_model import *
from src.activationf.relu import relu
from src.utils.normalize_data import normalize_data
from src.activationf.sigmoid import sigmaf
from config import monk_config

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
    early_stopping = [True],
    epsilon = [1e-5],
    patience = [20],
    momentum = [True],
    alpha_mom = [0.9],
    split = [0.2]
)

best_config, best_mee = gs.run(tr_input, tr_target, scouting_epochs=100)
print(" 🏆🚀 BEST CONFIG: \n", best_config, "\n\n\n", "BEST MEE: ", best_mee)
"""

# Avvia training
trainer = Trainer(tr_input.shape[1],
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

if (monk_config.EARLY_STOPPING == True):
    mee_tr, mse_tr, mee_vl, mse_vl =trainer.fit(tr_input, tr_target, 
                             vl_input, vl_target)
else:


    trainer.fit(tr_input, tr_target, vl_input, vl_target)

