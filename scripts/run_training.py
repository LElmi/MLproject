from src.training import trainer
from src.training.trainer import Trainer

from src.utils.load_data import load_data

import config

# Carico dati
x_i, d = load_data("data/training_data/ML-CUP25-TR.csv")
x_i = x_i.to_numpy()
d = d.to_numpy()

# Creo trainer
trainer = Trainer(x_i.shape[1],
                  config.N_HIDDENL1,
                  config.N_HIDDENL2,
                  config.N_OUTPUTS,
                  config.FUN_ACT,
                  config.LEARNING_RATE,
                  config.EPOCHS)

# Avvio training (grazie a __call__)
trainer.train(x_i, d)
