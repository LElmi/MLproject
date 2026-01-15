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
    units_list = [[32,16],[64, 32], [64, 128, 32]],
    n_outputs = [4],
    f_act_hidden = [relu],
    f_act_output = [linear],
    learning_rate = [0.0001, 0.001],
    use_decay = [True, False],
    decay_factor = [0.99, 0.95],
    decay_step = [100],
    batch = [True],
    epochs = [100],
    early_stopping = [True],
    epsilon = [1e-5],
    patience = [10],
    momentum = [True],
    alpha_mom = [0.9, 0.6],
    max_gradient_norm = [5],
    split = [0.2]
)

best_config, best_mee = gs.run(tr_input, tr_target, scouting_epochs=100)
print(" 🏆🚀 BEST CONFIG: \n", best_config, "\n\n\n", "BEST MEE: ", best_mee)

# |||| TRAIN FINALE SULLA MIGLIORE CONFIGURAZIONE TROVATA ||||

best_config["epochs"] = 15000
best_config["epsilon"] = 1e-7

train_best_config = Trainer(
    input_size = tr_input.shape[1],
    **best_config,
    verbose = True,
    validation = True
)

train_best_config.fit(tr_input, tr_target, vl_input, vl_target)

