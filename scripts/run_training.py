from src.training import trainer


from src.training.trainer import Trainer
from src.utils.load_data import load_data
from src.activationf.sigmoid import dsigmaf
from src.activationf.relu import relu_deriv

# Carico dati
x_i, d = load_data("data/training_data/ML-CUP25-TR.csv")
x_i = x_i.to_numpy()
d = d.to_numpy()

# Creo trainer
trainer = Trainer(x_i.shape[1])

# Avvio training (grazie a __call__)
trainer.train(x_i, d, dfunact=relu_deriv)
