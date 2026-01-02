import numpy as np
from src.training.trainer import Trainer
from src.utils.load_data import load_data, load_monks_data
from scripts.run_validation import *
import copy
best_model = None
best_accuracy = float("inf")
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

for seed in range(10):
    np.random.seed(seed)
    trainer = Trainer(x_i_remaining.shape[1],
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
                              config.LAMBDA)
    weights_filename, architecture_filename=trainer.train_with_early_stopping(x_i_remaining, d, validation_set, validation_d)
    w_j1i, w_j2j1, w_kj2, architecture = load_model(weights_filename, architecture_filename)
    correctly_classified, misclassified=validation_monk(validation_set, validation_d, w_j1i, w_j2j1, w_kj2)
    accuracy=correctly_classified/(correctly_classified + misclassified)
    if accuracy < best_accuracy:
        best_accuracy = accuracy
        best_model = copy.deepcopy(trainer)
    print("miglior accuratezza raggiunta:",best_accuracy*100,"%")
