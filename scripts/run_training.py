from src.training import trainer
from src.training.trainer import Trainer

from src.utils.load_data import load_data

import config

# Carica dati
x_i, d = load_data(config.PATH_DT)
x_i = x_i.to_numpy()
d = d.to_numpy()

# Inizializza la classe Trainer:
#  - crea la rete neurale
#  - traccia gli errori
#  - implementa metodo che esegue il training

trainer = Trainer(x_i.shape[1],
                  config.N_HIDDENL1,
                  config.N_HIDDENL2,
                  config.N_OUTPUTS,
                  config.FUN_ACT,
                  config.LEARNING_RATE,
                  config.BATCH,
                  config.EPOCHS,
                  config.EPSILON,
                  config.PATIENCE,
                  config.MOMENTUM,
                  config.ALPHA_MOM)

# normalizza le matrici x_i e d in l2
min_val_x = x_i.min(axis=0)
max_val_x = x_i.max(axis=0)
x_i = (x_i - min_val_x) / (max_val_x - min_val_x)

min_val_d = d.min(axis=0)
max_val_d = d.max(axis=0)
d = (d - min_val_d) / (max_val_d - min_val_d)


# Avvia training
trainer.train(x_i, d)
