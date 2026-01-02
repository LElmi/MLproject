from src.training.train.trainer import Trainer
from src.utils.normalize_data import normalize_data
from src.utils.load_data import load_data

import config
import numpy as np

x_i, d = load_data(config.PATH_DT)
x_i = x_i.to_numpy()
d = d.to_numpy()

# Normalizza Input
x_i, x_min, x_max = normalize_data(x_i)

# Normalizza Target (Fondamentale!)
d, d_min, d_max = normalize_data(d)

# Inizializza la classe Trainer:
#  - crea la rete neurale
#  - traccia gli errori
#  - implementa metodo che esegue il training

trainer = Trainer(
    input_size=x_i.shape[1],
    units_list=config.UNITS_LIST,
    n_outputs=config.N_OUTPUTS,
    f_act=config.FUN_ACT,
    learning_rate=config.LEARNING_RATE,
    use_decay=config.USE_DECAY,
    decay_factor= config.DECAY_FACTOR,
    decay_step=config.DECAY_STEP,
    batch=config.BATCH,
    epochs=config.EPOCHS,              # <--- PRIMA epochs
    early_stopping=config.EARLY_STOPPING, # <--- POI early_stopping
    epsilon=config.EPSILON,
    patience=config.PATIENCE,
    momentum=config.MOMENTUM,
    alpha_mom=config.ALPHA_MOM,
    split=config.SPLIT,
    verbose=True
)

# Avvia training
trainer.fit(x_i, d)

"""# 1. Ottieni le predizioni finali della rete per tutto il dataset
predictions_norm = []
for x in x_i:
    res = trainer.neuraln.forward_network(x)
    predictions_norm.append(res[-1])
predictions_norm = np.array(predictions_norm)

# 2. Applica la formula inversas
# Formula: ValoreReale = ValoreNorm * (Max - Min) + Min
predictions_real = predictions_norm * (d_max - d_min) + d_min
targets_real = d * (d_max - d_min) + d_min

# 3. Calcola il MEE reale finale
errors = targets_real - predictions_real
mee_reale = np.mean(np.sqrt(np.sum(errors**2, axis=1)))

print(f"MEE Finale sui dati REALI: {mee_reale:.4f}")"""
